#!/usr/bin/env python

from ..builtins import built_in_functions
from ..symboltable import scalar_types, array_types
from ..symboltable import SymbolTable, Scope, Keyword, ArrayElement, Variable, Window, \
                          Array, ParallelFunction, SequentialFunction, NeutralFunction, DisplayWindow
from ..nodes import *
from ..node_helpers import indent, scoped, parallel_scoped, check_is_symbol, check_type, \
                           collect_elements_from_list, list_contains_type, list_contains_function
from ..conversions import c_type_from_chestnut_type, color_properties, coordinates, array_to_array_element
from ..exceptions import CompilerException

from collections import namedtuple, defaultdict

from jinja2 import Environment, PackageLoader
jinja = Environment(loader=PackageLoader('chestnut', 'templates'))

def rendered(template_name, **kwargs):
    template = jinja.get_template(template_name)
    return template.render(**kwargs)

# special magic that should probably be in WalnutBackend but I can't figure out
# the magic python incantations to put it there without errors
code_generators = dict()
def generates_code_for(node_type):
    def wrapper(f):
        code_generators[node_type] = f
        return f

    return wrapper

function_code_generators = dict()
def generates_code_for_function(function_name):
    def wrapper(f):
        function_code_generators[function_name] = f
        return f

    return wrapper


class WalnutBackend(object):
    class GeneratorException(Exception): pass

    def __init__(self):
        self.symbols = None
        self.code = None

        # This is hacky and will break if WalnutBackend is not a singleton
        jinja.filters['cpp'] = self.code_for
        jinja.filters['cpp_list'] = self.code_for_nodes

    def code_for(self, node):
        try:
            generate = code_generators[node.__class__]
        except AttributeError:
            raise GeneratorException("No generator exists for nodes with type: {0}".format(type(node)))

        return generate(self, node)

    def code_for_nodes(self, nodes):
        return tuple(map(self.code_for, nodes))

    def code_for_function(self, function_node):
        try:
            generate = function_code_generators[function_node.name]
        except AttributeError:
            raise GeneratorException("No generator exists for function with name: {0}".format(function_node.name))

        return generate(self, function_node)

    def compile(self, ast):
        self.symbols = SymbolTable()
        self.code = defaultdict(list)
        self.ast = ast

        for function in built_in_functions:
            self.symbols.add(function)

        return self.code_for(ast)


    ####### Generators #######
    @generates_code_for(Program)
    def Program(self, node):
        functions = []
        statements = []

        # allow globals to be overridden
        self.symbols.createScope()

        for statement in node:
            if statement.__class__ in [SequentialFunctionDeclaration, ParallelFunctionDeclaration]:
                functions.append(self.code_for(statement))
                statements.append("// function '%s' was declared (see above)" % statement.name)
            else:
                statements.append(self.code_for(statement))

        self.symbols.removeScope()

        return rendered('walnut/base.cpp',
                        symbolTable=self.symbols,
                        main_statements=statements,
                        declarations=self.symbols.parallelFunctionDeclarations + self.symbols.sequentialFunctionDeclarations,
                        functions=functions)

    @generates_code_for(VariableDeclaration)
    def VariableDeclaration(self, node):
        self.symbols.add(Variable(node.name, node.type))


        code = '{type} {name};'.format(type=self.code_for(node.type), name=self.code_for(node.name))
        if node.initialization:
            code += ' ' + self.code_for(Statement([Assignment([node.name, node.initialization.expression])]))

        return code


    @generates_code_for(ArrayDeclaration)
    def ArrayDeclaration(self, node):
        self.symbols.add(Array(node.name, node.type, node.size))

        t = jinja.from_string("Array<{{ type }}> {{ name }} = _allocator.arrayWithSize<{{ type }}>({{ dimensions|join(', ') }});")
        code = t.render(type=self.code_for(node.type),
                        name=self.code_for(node.name),
                        dimensions=node.size.children)


        if node.initialization:
            code += ' ' + self.code_for(Statement([Assignment([node.name, node.initialization.expression])]))

        return code

    @generates_code_for(SequentialFunctionDeclaration)
    def SequentialFunctionDeclaration(self, node):
        template_declaration = jinja.from_string("__host__ {{type}} {{name}}({{parameters|cpp_list|join(', ')}});")
        template_definition = jinja.from_string("__host__ {{type}} {{name}}({{parameters|cpp_list|join(', ')}}) {{block|cpp}}")

        self.symbols.add(SequentialFunction(node.name, node.type, node.parameters, node=node))
        self.symbols.createScope()


        for parameter in node.parameters:
            if parameter.type in scalar_types:
                self.symbols.add(Variable(name=parameter.name, type=parameter.type))
            if parameter.type in array_types:
                self.symbols.add(Array(name=parameter.name, type=parameter.type, size=None))

        environment = { 'name' : node.name,
                        'type' : node.type,
                        'parameters' : node.parameters,
                        'block' : node.block }

        declaration = template_declaration.render(**environment)
        function = template_definition.render(**environment)

        self.symbols.sequentialFunctionDeclarations.append(declaration)

        self.symbols.removeScope()
        return function

    @generates_code_for(ParallelFunctionDeclaration)
    def ParallelFunctionDeclaration(self, node):
        template_declaration = jinja.from_string("__device__ {{type}} {{name}}({{parameters|cpp_list|join(', ')}});")
        template_definition = jinja.from_string("__device__ {{type}} {{name}}({{parameters|cpp_list|join(', ')}}) {{block|cpp}}")

        if list_contains_type(node.block, ParallelContext):
            raise CompilerException("Parallel contexts (foreach loops) can not exist inside of parallel functions", node)

        self.symbols.add(ParallelFunction(node.name, node.type, node.parameters, node=node))
        self.symbols.createParallelScope()

        for parameter in node.parameters:
            self.symbols.add(Variable(name=parameter.name, type=parameter.type))

        environment = { 'name' : node.name,
                        'type' : node.type,
                        'parameters' : node.parameters,
                        'block' : node.block }

        declaration = template_declaration.render(**environment)
        function = template_definition.render(**environment)

        self.symbols.parallelFunctionDeclarations.append(declaration)

        self.symbols.removeScope()
        return function


    @generates_code_for(Type)
    def Type(self, node):
        return c_type_from_chestnut_type(node)

    @generates_code_for(Symbol)
    def Symbol(self, node):
        # TODO: Potentially change symbols here in some way to make them not
        # clash with other walnut/cuda stuff
        symbol = self.symbols.lookup(str(node))
        if not symbol:
            raise CompilerException("Symbol %s has not been declared" % node) #TODO... make symbols ChetsnutNodes
        try:
            return symbol.cpp_name
        except AttributeError:
            return symbol.name

    @generates_code_for(String)
    def String(self, node):
        return 'UNIMPLEMENTED'

    @generates_code_for(Integer)
    def Integer(self, node):
        return str(node)

    @generates_code_for(Real)
    def Real(self, node):
        return str(node)

    @generates_code_for(Bool)
    def Bool(self, node):
        return str(node._value).lower()

    ### Code for !, +, -, *, /, %
    @generates_code_for(Not)
    def Not(self, node):
        return '!%s' % self.code_for(node.operand)
    @generates_code_for(Neg)
    def Neg(self, node):
        return '(-%s)' % self.code_for(node.operand)

    @generates_code_for(Product)
    def Product(self, node):
        return jinja.from_string("({{ factors|join('*') }})").render(factors=self.code_for_nodes(node.children))
    @generates_code_for(Inverse)
    def Inverse(self, node):
        return jinja.from_string("(1.0/{{ divisor }})").render(divisor=self.code_for(node.divisor))

    @generates_code_for(Sum)
    def Sum(self, node):
        return jinja.from_string("({{ terms|join('+') }})").render(terms=self.code_for_nodes(node.children))

    @generates_code_for(Mod)
    def Mod(self, node):
        return jinja.from_string("(({{ left }}) % ({{ right }}))").render(left=node.left, right=node.right)

    ### Code for <, <=, >, >=, ==, !=
    @generates_code_for(LessThan)
    def LessThan(self, node):
        return '({0} < {1})'.format(*self.code_for_nodes(node.children))
    @generates_code_for(LessThanOrEqual)
    def LessThanOrEqual(self, node):
        return '({0} <= {1})'.format(*self.code_for_nodes(node.children))

    @generates_code_for(GreaterThan)
    def GreaterThan(self, node):
        return '({0} > {1})'.format(*self.code_for_nodes(node.children))
    @generates_code_for(GreaterThanOrEqual)
    def GreaterThanOrEqual(self, node):
        return '({0} >= {1})'.format(*self.code_for_nodes(node.children))

    @generates_code_for(Equal)
    def Equal(self, node):
        return '({0} == {1})'.format(*self.code_for_nodes(node.children))
    @generates_code_for(NotEqual)
    def NotEqual(self, node):
        return '({0} != {1})'.format(*self.code_for_nodes(node.children))

    ### Code for &&, ||
    @generates_code_for(BooleanAnd)
    def BooleanAnd(self, node):
        return '({0} && {1})'.format(*self.code_for_nodes(node.children))

    @generates_code_for(BooleanOr)
    def BooleanOr(self, node):
        return '({0} || {1})'.format(*self.code_for_nodes(node.children))

    ### Code for =, +=, -=, *=, /=
    @generates_code_for(Assignment)
    def Assignment(self, node):
        #TODO: I'd like to use this approach for the rest of the assigners
        # but it seems to not be universally applicable because we have no way
        # of knowing what type of assignment (=, +=, etc..) it was.
        try:
            self.symbols.insert('@variable_to_assign', self.symbols.lookup(node.variable))
            return '{0} = {1}'.format(*self.code_for_nodes(node.children))
        finally:
            self.symbols.remove('@variable_to_assign')

    @generates_code_for(AssignPlus)
    def AssignPlus(self, node):
        return '{0} += {1}'.format(*self.code_for_nodes(node.children))

    @generates_code_for(AssignMinus)
    def AssignMinus(self, node):
        return '{0} -= {1}'.format(*self.code_for_nodes(node.children))

    @generates_code_for(AssignMul)
    def AssignMul(self, node):
        return '{0} *= {1}'.format(*self.code_for_nodes(node.children))

    @generates_code_for(AssignDiv)
    def AssignDiv(self, node):
        return '{0} /= {1}'.format(*self.code_for_nodes(node.children))

    # Code for 
    @scoped
    @generates_code_for(Block)
    def Block(self, node):
        return rendered('walnut/block.cpp',
                        statements=self.code_for_nodes(node.children))

    @generates_code_for_function('display')
    def ArrayDisplay(self, node):
        node = node.expressions.convert_to(ArrayDisplay)

        display_functor_template = jinja.from_string("""\
template <typename InputType>
struct _color_convert__{{display_function_name}} : public thrust::unary_function<{{input_data_type}}, Color> {
    __device__
    Color operator()({{input_data_type}} value) {
        return {{display_function_name}}(value);
    }
};
""")
        display_template = jinja.from_string("""\
{
  thrust::copy(thrust::make_transform_iterator({{input}}.thrustPointer(), {{display_function}}<{{template_types}}>()),
               thrust::make_transform_iterator({{input}}.thrustEndPointer(), {{display_function}}<{{template_types}}>()),
               _{{input}}_display.displayData());
}
_{{input}}_display.updateGL();
_app.processEvents();
""")

        data = self.symbols.lookup(node.data)

        if node.display_function:
            display_function = self.symbols.lookup(node.display_function)
            check_type(display_function, ParallelFunction)
            if display_function.type != 'Color':
                raise CompilerException("Display function '%s' returns data of type '%s'. Display functions must return 'Color'."
                        % (display_function.name, display_function.type))
            parameters = display_function.parameters
            if len(parameters) != 1:
                raise CompilerException("Display function '%s' takes %d parameters. Display functions must only take 1 parameter"
                        % (display_function.name, len(parameters)))
            #if data.type != parameters[0].type:
            #    raise CompilerException("Display function takes a parameter of type '%s' but the data '%s' is of the type '%s'"
            #            % (parameters[0].type, data.name, data.type))


            functor = display_functor_template.render(display_function_name=display_function.name,
                                                      input_data_type=c_type_from_chestnut_type(data.type))

            if functor not in self.symbols.parallelContexts:
                self.symbols.parallelContexts.append(functor)

            #TODO: There's got to be a better way to do this color conversion stuff
            display_function = '_color_convert__{0}'.format(display_function.name)
        else:
            display_function = '_chestnut_default_color_conversion'

        template_types = str(data.type)

        display_env = { 'input' : data.name,
                        'template_types' : template_types,
                        'display_function' : display_function }

        code = ""
        if not data.has_display:
            data.has_display = True
            self.symbols.displayWindows.append(DisplayWindow(data.name, 'Chestnut Output [%s]' % data.name, data.size.width, data.size.height))
        code += display_template.render(**display_env)
        return code


    @generates_code_for_function('print')
    def Print(self, node):
        node = node.expressions.convert_to(Print)
        num_format_placeholders = node.format_string.count('%s')
        num_args = len(node.children)-1

        if num_format_placeholders != num_args:
            raise CompilerException("There are %s format specifiers but %s arguments to print()" % \
                                    (num_format_placeholders, num_args), node)

        format_substrings = node.format_string.split('%s')

        code = 'std::cout'
        for substring, placeholder in zip(format_substrings[:-1], node.children[1:]):
            code += ' << "%s"' % substring
            code += ' << %s' % self.code_for(placeholder)
        code += ' << "%s" << std::endl' % format_substrings[-1]

        return code


    @generates_code_for(FunctionCall)
    def FunctionCall(self, node):
        template = jinja.from_string("{{name|cpp}}({{arguments|cpp_list|join(', ')}})")

        # use a built in generator for this function if we have one
        if node.name in function_code_generators:
            return self.code_for_function(node)

        check_is_symbol(node.name, self.symbols)
        function = self.symbols.lookup(node.name)

        if self.symbols.in_parallel_context:
            check_type(function, ParallelFunction, NeutralFunction)
        else:
            check_type(function, SequentialFunction, NeutralFunction)

        if not len(node.expressions) == len(function.parameters):
            raise CompilerException("The function %s(%s) takes %s parameters but %s were given" % \
                    (function.name, ', '.join(map(lambda x: x[1], function.parameters)), len(function.parameters), len(node.expressions)), node)

        return template.render(name=node.name, arguments=node.expressions)


    @generates_code_for(Size)
    def Size(self, node):
        return 'UNIMPLEMENTED'

    @generates_code_for(Parameter)
    def Parameter(self, node):
        # we don't do type|cpp and name|cpp here because the symbols haven't been added to the symbol table
        return jinja.from_string('{{type}} {{name}}').render(name=node.name, type=node.type)

    @generates_code_for(Statement)
    def Statement(self, node):
        return jinja.from_string('{{expression|cpp}};').render(expression=node.expression)

    @generates_code_for(Property)
    def Property(self, node):
        property_template = jinja.from_string("{{name}}.{{property}}({{parameters}})")
        rest = node.children[1:]

        # New property pseudo code
        property_cpp = self.code_for(node.expression)

        # HACK: There should be a general property framework
        if type(self.symbols.lookup(node.expression)) == ArrayElement and rest[0] in color_properties:
            property_cpp += '.center()'

        for member_identifier in rest:
            #check_that_symbol_type_has_member(current_symbol_type, member_identifier) # need to implement
            #current_symbol_type = get_return_type_for_member(current_symbol_type, member_identifier) # need to implement

            property_cpp = property_template.render(**{ 'name' : property_cpp,
                                                        'property' : member_identifier,
                                                        'parameters' : '' })

        return property_cpp

        #if type(symbol) == Variable:
        #    if symbol.type == 'Color':
        #        return '%s.%s' % (symbol.name, color_properties[property_])
        #    elif symbol.type == 'Point2d' and property_ in ['x', 'y']:
        #        return '%s.%s' % (symbol.name, property_)
        #    elif symbol.type == 'Size2d' and property_ in ['width', 'height']:
        #        return '%s.%s' % (symbol.name, property_)
        #    elif symbol.type in ['IntWindow2d', 'RealWindow2d', 'ColorWindow2d'] \
        #       and property_ in ['topLeft', 'top', 'topRight', 'left', 'center', 'right', 'bottomLeft', 'bottom', 'bottomRight']:
        #        return property_template % { 'name' : name,
        #                                    'property' : property_,
        #                                    'parameters' : '' }
        #    else:
        #        raise Exception('unknown property variable type %s.%s' % (symbol.type, property_))

        #elif type(symbol) == Array:
        #    if property_ not in ['size']:
        #        raise CompilerException("Error, property '%s' not part of the data '%s'" % (property_, name))
        #    return property_template % { 'name' : name,
        #                                 'property' : property_,
        #                                 'parameters' : '' }

        #else:
        #    print symbolTable
        #    print self
        #    raise Exception('unknown property type')

    @generates_code_for(ArrayReference)
    def ArrayReference(self, node):
        references = node.children[1:]

        check_is_symbol(node.identifier, self.symbols)
        symbol = self.symbols.lookup(node.identifier)

        if type(symbol) == Array:
            # TODO: Support 1d and 3d. Assuming 2d for now.
            if len(references) > 3:
                raise CompilerException("Too many arguments to the array lookup, there should only be between 1 and 3 (i.e. %s[x, y, z])" % symbol.name, node)
            elif len(references) < 1:
                raise CompilerException("Too few arguments to the array lookup, there should be between 1 and 3 (i.e. %s[x, y, z])" % symbol.name, node)
            #check_expression_type(arg1)
            #check_expression_type(arg2)

            template = jinja.from_string("{{name}}.at({{references|cpp_list|join(', ')}})")
            return template.render(name=symbol.cpp_name, references=references)

        else:
            raise CompilerException("%s should be an array if you want to do %s[x, y, z] lookups on it" % symbol.name, node)

    @generates_code_for(Return)
    def Return(self, node):
        return jinja.from_string('return {{expression|cpp}};').render(expression=node.expression)

    @generates_code_for(Break)
    def Break(self, node):
        return 'break;'

    @generates_code_for(If)
    def If(self, node):
        template = jinja.from_string("if ({{node.condition|cpp}}) {{node.if_statement|cpp}} {% if node.else_statement %}else {{node.else_statement|cpp}}{% endif %}")

        return template.render(node=node)

    @generates_code_for(While)
    def While(self, node):
        return jinja.from_string("while ({{node.condition|cpp}}) {{node.statement|cpp}}").render(node=node)

    @generates_code_for(For)
    def For(self, node):
        for_template = jinja.from_string("""\
// FOR LOOP
{{node.initialization|cpp}};
while ({{node.condition|cpp}}) {
  {{node.statement|cpp}}
  {{node.step|cpp}};
}
// END FOR LOOP
""")
        return for_template.render(node=node)

    @generates_code_for_function('index')
    def ParallelIndex(self, node):
        return '_index'

    @generates_code_for_function('random')
    def Random(self, node):
        node = node.convert_to(Random)

        random_device_template = "walnut_random()"

        if self.symbols.in_parallel_context:
            return random_device_template
        else:
            raise CompilerException("Sequential Random not implemented")

    @generates_code_for_function('reduce')
    def ArrayReduce(self, node):
        node = node.convert_to(ArrayReduce)
        return 'UNIMPLEMENTED'
#reduce_template = """\
#thrust::reduce({input_data}.thrustPointer(), {input_data}.thrustEndPointer())\
#"""
#class ArrayReduce(ChestnutNode):
#    @property
#    def input(self): return self[0]
#
#    @property
#    @safe_index
#    def function(self): return self[1]
#
#    def to_cpp(self, env=defaultdict(bool)):
#        function = None
#
#        check_is_symbol(self.input)
#        input = self.symbols.lookup(self.input)
#
#        check_type(input, Array)
#
#        #TODO: Actually use function
#        if self.function:
#            print('Warning: functions in reduce() calls are not implemented yet')
#            check_is_symbol(self.function)
#            function = self.symbols.lookup(self.function)
#            check_type(self.function, SequentialFunction)
#            return ''
#
#        else:
#            return reduce_template.format(input_data=input.name)

    @generates_code_for_function('sort')
    def ArraySort(self, node):
        node = node.convert_to(ArraySort)

        sort_template = jinja.from_string("""{{input_data}}.sorted_copy_in({{output_data}})""")

        output = self.symbols.lookup('@variable_to_assign')
        if not output:
            raise InternalException("The symbol table doesn't have a '@variable_to_assign' variable set.\n%s" % self.symbols)

        check_is_symbol(self.input)
        check_is_symbol(output)

        input = self.symbols.lookup(self.input)
        output = self.symbols.lookup(output)

        check_type(input, Array)
        check_type(output, Array)

        #TODO: Actually use function
        if node.function:
            print('Warning: functions in reduce() calls are not implemented yet')
            check_is_symbol(node.function)
            function = self.symbols.lookup(node.function)
            check_type(function, SequentialFunction)
            return ''

        else:
            return sort_template % { 'input_data' : input.name,
                                     'output_data' : output.name }

    @generates_code_for_function('read')
    def ArrayRead(self, node):
        node = node.convert_to(ArrayRead)
        return '{data}.readFromFile("{filename}")'.format(data=self.symbols.lookup('@variable_to_assign').name, filename=node.filename)

    @generates_code_for_function('write')
    def ArrayWrite(self, node):
        node = node.convert_to(ArrayWrite)
        return '{data}.writeToFile("{filename}")'.format(data=node.data, filename=node.filename)

    @generates_code_for(ForeachParameter)
    def ForeachParameter(self, node):
        return 'UNIMPLEMENTED'

    @generates_code_for(ForeachParameter)
    def ForeachParameter(self, node):
        return 'UNIMPLEMENTED'

    @generates_code_for(ParallelContext)
    @parallel_scoped
    def ParallelContext(self, node):
        # TODO: eww, this looks ugly. Can't I make it better somehow while
        # still keeping templates close to the code that uses them?
        parallel_context_declaration = jinja.from_string("""\
struct {{function_name}} {
    {{struct_members}}
    {{function_name}}({{function_parameters}}) {{struct_member_initializations}} {}

    template <typename Tuple>
    __device__
    void operator()(Tuple _t) {{function_body}}
};
""")
        parallel_context_call = jinja.from_string("""\
{
    FunctionIterator iterator = makeStartIterator({{size_name}});

    {{temp_arrays}}

    thrust::for_each(iterator, iterator+{{size_name}}.length(), {{function}}({{variables}}));

    {{swaps}}
    {{releases}}
}
""")

        #TODO: Clean this shit up!

        context_name = '_foreach_context_' + self.symbols.uniqueNumber()
        arrays = [] # allows us to check if we're already using this array under another alias
        outputs = []
        requested_variables = [] # Allows for function calling

        struct_member_variables = []
        function_parameters = []
        struct_member_initializations = []

        copies = ''

        # Add the Array and ArrayElement to the current scope
        for parameter in node.parameters:
            if parameter.array not in arrays:
                arrays.append(parameter.array)
            else:
                raise CompilerException("{name} already used in this foreach loop".format(name=parameter.array), node)

            array = self.symbols.lookup(parameter.array)
            if array.type not in array_types:
                raise CompilerException("%s %s should be an array" % (array.type, array.name), node)


            self.symbols.add(ArrayElement(name=parameter.array_element,
                                            type=c_type_from_chestnut_type(array.type),
                                            array=array,
                                            #TODO: templatize
                                            cpp_name='%s(%s, _x, _y, _z)' % (array_to_array_element[array.type], array.name)))

            self.symbols.add(array)

            #TODO: Templatize
            struct_member_variables.append('%s %s;' % (array.type, array.name))
            function_parameters.append('%s _%s' % (array.type, array.name))
            struct_member_initializations.append('%(name)s(_%(name)s)' % { 'name' : array.name })
            requested_variables.append(array.name)

        #TODO: I could make this check lazy (check in the child parallel context/return statement
        # to speed up compilation by being optimistic
        if list_contains_type(node.statements, ParallelContext):
            raise CompilerException("Parallel Contexts (foreach loops) can not be nested")

        if list_contains_type(node.statements, Return):
            raise CompilerException("Parallel Contexts (foreach loops) can not contain return statments")


        # Look at all of the symbols and see if they need to be passed into the context
        for variable in collect_elements_from_list(node.statements, Symbol):
            if variable not in self.symbols.currentScope and \
               self.symbols.lookup(variable) and \
               self.symbols.lookup(variable).name not in requested_variables and \
               type(self.symbols.lookup(variable)) not in [ParallelFunction, SequentialFunction, NeutralFunction]:
                # We've got a variable that needs to be passed into this function
                variable = self.symbols.lookup(variable)
                if variable.type in array_types:
                    struct_member_variables.append('%s %s;' % (variable.type, variable.name))
                    function_parameters.append('%s _%s' % (variable.type, variable.name))
                    struct_member_initializations.append('%(name)s(_%(name)s)' % { 'name' : variable.name })
                else:
                    struct_member_variables.append('%s %s;' % (variable.type, variable.name))
                    function_parameters.append('%s _%s' % (variable.type, variable.name))
                    struct_member_initializations.append('%(name)s(_%(name)s)' % { 'name' : variable.name})

                requested_variables.append(variable.name)

        for assignment in collect_elements_from_list(node.statements, Assignment):
            if type(assignment.variable) == Property:
                continue
            symbol = self.symbols.lookup(assignment.variable)
            if symbol and type(symbol) == ArrayElement and symbol not in outputs:
                outputs.append(symbol)

                struct_member_variables.append('{0.type} _original_values_for_{0.name};'.format(symbol.array))
                function_parameters.append('{0.type} __original_values_for_{0.name}'.format(symbol.array))
                struct_member_initializations.append('_original_values_for_%(name)s(__original_values_for_%(name)s)' % { 'name' : symbol.array.name })

                copies += '%s = %s;\n' % (symbol.cpp_name, '{1}(_original_values_for_{0.name}, _x, _y, _z)'.format(symbol.array, array_to_array_element[symbol.array.type]))


        new_names = []

        for (new_name, original_name) in [('_new_values_for_%s' % output.array.name, output.array.name) for output in outputs]:
            #Swap order of new and old so that it matches the kernel order. Weird, I know
            requested_variables[requested_variables.index(original_name)] = new_name
            requested_variables.append(original_name)
            new_names.append(new_name)

        temp_arrays = [self.code_for(ArrayDeclaration([output.array.type, Symbol(name), output.array.size]))
                       for output, name in zip(outputs, new_names)]

        data_swap_template = """\
        {first_data}.swapDataWith({second_data});
        """
        swaps = map(lambda (a, b): data_swap_template.format(first_data=a, second_data=b.array.name), zip(new_names, outputs))

        data_release_template = """\
        _allocator.releaseArray({name});
        """

        releases = [data_release_template.format(name=temp) for output, temp in zip(outputs, new_names)]

        if len(node.parameters) > 0:
            struct_member_variables = indent(indent('\n'.join(struct_member_variables), indent_first_line=False), indent_first_line=False)
            function_parameters = ', '.join(function_parameters)
            struct_member_initializations = ": " + ', '.join(struct_member_initializations)
        else:
            struct_member_variables = function_parameters = struct_member_initializations = ''

        if list_contains_function(node.statements, 'random'): #TODO: fix random
            random_initializer = "curandState __random_state;\ncurand_init(1234 /* seed */, hash(_index), 0, &__random_state);\n"
        else:
            random_initializer = ''

        body = '{\n' + copies + indent(random_initializer + '\n'.join(self.code_for_nodes(node.statements))) + '\n}'
        body = indent(indent(body, indent_first_line=False), indent_first_line=False) # indent twice

        environment = { 'function_name' : context_name,
                        'function_body' : body,
                        'struct_members' : struct_member_variables,
                        'function_parameters' : function_parameters,
                        'struct_member_initializations' : struct_member_initializations }

        function_declaration = parallel_context_declaration.render(**environment)


        # Function Call
        size = self.symbols.lookup(node.parameters[0].children[1]).name + '.size()'


        function_call = parallel_context_call.render(**{ 'variables' : ', '.join(requested_variables),
                                         'temp_arrays' : indent(''.join(temp_arrays), indent_first_line=False),
                                         'swaps' : indent(''.join(swaps), indent_first_line=False),
                                         'releases' : indent(''.join(releases), indent_first_line=False),
                                         'function' : context_name,
                                         'size_name' : size })


        self.symbols.parallelContexts.append(function_declaration)

        return function_call

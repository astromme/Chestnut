from lepl import List

class ParseContext(List):
    @property
    def start_line(self):
        return self[0]
    @start_line.setter
    def start_line(self, value):
        self[0] = value

    @property
    def end_line(self):
        return self[1]
    @end_line.setter
    def end_line(self, value):
        self[1] = value

# The base node for more complex chestnut features
class ChestnutNode(List):
    child_names = []

    def __init__(self, *args, **kwargs):
        super(ChestnutNode, self).__init__(*args, **kwargs)

        # if we don't have a context, create a fake one to keep the AST API consistent
        if len(self) == 0 or type(self[0]) != ParseContext:
            self.insert(0, ParseContext([None, None]))

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        for child in self.children:
            yield child

    # map child_names onto the list structure to allow for node.blah
    def __getattr__(self, name):
        if name in self.child_names:
            try:
                # context always in slot 0
                index = self.child_names.index(name)+1
                #print name, "has index", index
                return self[index]
            except IndexError:
                return None

    @property
    def children(self):
        return self[1:]

    # provide access to the context
    @property
    def context(self):
        assert type(self[0]) == ParseContext
        return self[0]
    @context.setter
    def context(self, value):
        self[0] = value

    def convert_to(self, node_type):
        node = node_type(self.children)
        node.context = self.context
        return node

class Type(str): pass
class Symbol(str): pass

class String(str): pass
class Integer(int): pass
class Real(float): pass
class Bool(object):
    def __init__(self, value):
        self._value = bool(value == 'yes' or value == 'true')
    def __nonzero__(self):
        return self._value
    def __repr__(self):
        return str(self._value)

class Not(ChestnutNode):                               child_names = ['operand']
class Neg(ChestnutNode):                               child_names = ['operand']

class Product(ChestnutNode): pass
class Inverse(ChestnutNode):                           child_names = ['divisor']

class Sum(ChestnutNode): pass

class Mod(ChestnutNode):                               child_names = ['left', 'right']

class LessThan(ChestnutNode): pass
class LessThanOrEqual(ChestnutNode): pass
class GreaterThan(ChestnutNode): pass
class GreaterThanOrEqual(ChestnutNode): pass
class Equal(ChestnutNode): pass
class NotEqual(ChestnutNode): pass

class BooleanAnd(ChestnutNode): pass
class BooleanOr(ChestnutNode): pass

class Assignment(ChestnutNode):                          child_names = ['variable', 'expression']
class AssignPlus(ChestnutNode):                          child_names = ['variable', 'expression']
class AssignMinus(ChestnutNode):                         child_names = ['variable', 'expression']
class AssignMul(ChestnutNode):                           child_names = ['variable', 'expression']
class AssignDiv(ChestnutNode):                           child_names = ['variable', 'expression']


class Program(List): pass
class VariableDeclaration(ChestnutNode):                 child_names = ['type', 'name', 'initialization']
class ArrayDeclaration(ChestnutNode):                    child_names = ['type', 'name', 'size', 'initialization']
class ObjectDeclaration(ChestnutNode):                   child_names = ['name', 'members', 'functions']
class SequentialFunctionDeclaration(ChestnutNode):       child_names = ['type', 'name', 'parameters', 'block']
class ParallelFunctionDeclaration(ChestnutNode):         child_names = ['type', 'name', 'parameters', 'block']
class Parameters(List): pass
class Block(ChestnutNode): pass
class VariableInitialization(ChestnutNode):              child_names = ['expression']
class ArrayInitialization(ChestnutNode):                 child_names = ['expression']
class ArrayDisplay(ChestnutNode):                        child_names = ['data', 'display_function']
class Print(ChestnutNode):                               child_names = ['format_string']
class FunctionCall(ChestnutNode):                        child_names = ['name', 'expressions']
class Size(ChestnutNode):                                child_names = ['width', 'height', 'depth']
class Parameter(ChestnutNode):                           child_names = ['type', 'name']
class Statement(ChestnutNode):                           child_names = ['expression']
class Expressions(ChestnutNode): pass
class Property(ChestnutNode):                            child_names = ['expression']
class ArrayReference(ChestnutNode):                      child_names = ['identifier']

class Return(ChestnutNode):                              child_names = ['expression']
class Break(ChestnutNode): pass

class If(ChestnutNode):                                  child_names = ['condition', 'if_statement', 'else_statement']
class While(ChestnutNode):                               child_names = ['condition', 'statement']
class For(ChestnutNode):                                 child_names = ['initialization', 'condition', 'step', 'statement']

class Random(ChestnutNode): pass
class ParallelLocation(ChestnutNode): pass
class ParallelWindow(ChestnutNode): pass
class ArrayReduce(ChestnutNode): pass
class ArraySort(ChestnutNode):                           child_names = ['input', 'function']
class ArrayRead(ChestnutNode):                           child_names = ['filename']
class ArrayWrite(ChestnutNode):                          child_names = ['data', 'filename']

class ForeachParameter(ChestnutNode):                    child_names = ['array_element', 'array']
class ParallelContext(ChestnutNode):                     child_names = ['parameters', 'statements']

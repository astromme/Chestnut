#!/usr/bin/env python

from jinja2 import Template, Environment, PackageLoader
jinja = Environment(loader=PackageLoader('chestnut', 'templates'))
base_template = jinja.get_template('base.cpp')

from .parser import parse
from .nodes import *
from .builtins import built_in_functions

function_types = [SequentialFunctionDeclaration, ParallelFunctionDeclaration]

def compile(ast):
    function_pairs = [func.to_cpp() for func in ast if type(func) in function_types]
    if len(function_pairs):
        declarations, functions = zip(*function_pairs)
    else:
        declarations, functions = [], []

    main_statements = [obj.to_cpp() for obj in ast if type(obj) not in function_types]

    return base_template.render(symbolTable=symbolTable,
                                main_statements=main_statements,
                                declarations=declarations,
                                functions=functions)

def main():
    import sys, os
    import shlex, subprocess

    for function in built_in_functions:
        symbolTable.add(function)

    if len(sys.argv) not in [2, 4]:
        print('Usage: %s input [-o output]' % sys.argv[0])
        sys.exit(1)

    input_file = sys.argv[1]
    splits = input_file.split('.')
    if splits[-1] not in ['chestnut', 'ch']:
        print("Sorry, your input file %s doesn't end with .chestnut or .ch" % input_file)

    if len(sys.argv) == 2:
        output_file = os.path.basename('.'.join(splits[0:-1]))
    else:
        output_file = sys.argv[3]

    try:
        with open(input_file, 'r') as f:
            code = ''.join(f.readlines())
    except IOError as e:
        print e
        sys.exit(1)

    # Time to get an AST, but that might fail and we want nice lines + context information
    try:
        ast = parse(code, from_file=input_file)
    except FullFirstMatchException, e:
        code_lines = code.split('\n')
        lineno = e.kargs['lineno']

        def valid_line(lineno):
            return lineno > 0 and lineno < len(code_lines)

        import textwrap
        print 'Error:'
        print '\n'.join(map(lambda s: '    ' + s, textwrap.wrap(str(e), 76)))
        print 'Context:'
        for offset in [-2, -1, 0, 1, 2]:
            if valid_line(lineno+offset):
                if offset == 0: print '  ->',
                else: print '    ',
                print '%s: ' % (lineno+offset) + code_lines[lineno+offset - 1]
        sys.exit(1)

    # Time to compile to C++ code, but that might fail and we want nice error messages
    try:
        thrust_code = compile(ast)
    except CompilerException as e:
        print e
        sys.exit(1)

    with open(output_file, 'w') as f:
      f.write(thrust_code)


if __name__ == '__main__':
  main()


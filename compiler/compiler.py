#!/usr/bin/env python

preamble = """\
#include <QApplication> // OpenGL stuff on Linux forces this to come first

#include <walnut/ArrayAllocator.h>
#include <walnut/Array.h>
#include <walnut/HostFunctions.h>
#include <walnut/UtilityFunctors.h>
#include <walnut/FunctionIterator.h>
#include <walnut/DisplayWindow.h>
#include <walnut/ColorKernels.h>

#include <walnut/Sizes.h>
#include <walnut/Points.h>
#include <walnut/Windows.h>
#include <walnut/Color.h>

#include <thrust/sort.h>

#include <limits.h>
#include <cutil.h>

using namespace Walnut;

ArrayAllocator _allocator;

"""

main_template = """

int main(int argc, char* argv[])
{
  %(app_statement)s

  // create gpu timer
  unsigned int _host_timer;
  cutCreateTimer(&_host_timer);
  cutResetTimer(_host_timer);

  srand(NULL);
  _allocator = ArrayAllocator();

%(main_declarations)s
%(main_code)s

  printf("time spent in kernels: %%0.2f ms\\n", cutGetTimerValue(_host_timer));

  %(return_statement)s
}
"""

from parser import parse
from nodes import *
from builtins import built_in_functions

for function in built_in_functions:
    symbolTable.add(function)

display_init = """\
DisplayWindow _%(name)s_display(QSize(%(width)s, %(height)s));
_%(name)s_display.show();
_%(name)s_display.setWindowTitle("%(title)s");
"""

function_types = [SequentialFunctionDeclaration, ParallelFunctionDeclaration]

def compile(ast):
    function_pairs = [func.to_cpp() for func in ast if type(func) in function_types]
    if len(function_pairs):
        declarations, functions = zip(*function_pairs)
    else:
        declarations, functions = [], []

    main_declarations = []
    main_function_statements = [obj.to_cpp() for obj in ast if type(obj) not in function_types]


    for window in symbolTable.displayWindows:
        main_declarations.append(display_init % { 'name' : window.name,
                                             'width' : window.width,
                                             'height' : window.height,
                                             'title' : window.title })


    if len(main_declarations):
      app_statement = 'QApplication _app(argc, argv);'
      return_statement = 'return _app.exec();'
    else:
      app_statement = ''
      return_statement = 'return 0;'

    main_function = main_template % { 'main_declarations' : indent('\n'.join(main_declarations)),
                                      'main_code' : indent('\n'.join(main_function_statements)),
                                      'app_statement' : app_statement,
                                      'return_statement' : return_statement }

    thrust_code = preamble + \
                  '\n'.join(symbolTable.structures) + \
                  '\n'.join(declarations) + \
                  '\n'.join(symbolTable.parallelContexts) + \
                  '\n'.join(functions) + main_function

    return thrust_code

def main():
    import sys, os
    import shlex, subprocess


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


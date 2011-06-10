#!/usr/bin/env python

preamble = """\
#include "walnut/DeviceData.h"
#include "walnut/DisplayWindow.h"
#include "walnut/ColorKernels.h"

#include <QApplication>
"""

main_template = """\

int main(int argc, char* argv[])
{
  %(app_statement)s

%(declarations)s
%(main_code)s

  %(return_statement)s
}
"""

from parser import parse
from nodes import *

cuda_directory='/usr/local/cuda/'
cuda_compiler=cuda_directory+'bin/nvcc'

cuda_compile_pass1 = """%(cuda_compiler)s -M -D__CUDACC__ %(input_file)s -o %(input_file)s.o.NVCC-depend -m64 -Xcompiler ,\"-g\" -DNVCC -I/usr/local/cuda/include -I/usr/local/cuda/include -I/usr/include -I /Library/Frameworks/QtGui.framework/Versions/4/Headers"""

cuda_compile_pass2 = """%(cuda_compiler)s %(input_file)s -c -o %(input_file)s.o -m64 -Xcompiler ,\"-g\" -DNVCC -I/usr/local/cuda/include -I/usr/local/cuda/include -I/usr/include -I /Library/Frameworks/QtGui.framework/Versions/4/Headers"""

cuda_compile_pass3 = """/usr/bin/c++ -O2 -g -Wl,-search_paths_first -headerpad_max_install_names  ./%(input_file)s.o -o %(output_file)s /usr/local/cuda/lib/libcudart.dylib -Wl,-rpath -Wl,/usr/local/cuda/lib /usr/local/cuda/lib/libcuda.dylib -L . -lwalnut -framework OpenGL -framework QtGui -framework QtOpenGL -framework QtCore"""


display_init = """\
ChestnutCore::DisplayWindow _%(name)s_display(QSize(%(width)s, %(height)s));
_%(name)s_display.show();
_%(name)s_display.setWindowTitle("%(title)s");
"""

def compile(ast):
  print ast
  functions = []
  declarations = []
  main_function_statements = []
  for program_node in ast:
    if type(program_node) in [SequentialFunctionDeclaration,
            ParallelFunctionDeclaration]:
      functions.append(program_node.to_cpp())
    elif type(program_node) == DataDeclaration:
      main_function_statements.append(program_node.to_cpp())
    elif type(program_node) == VariableDeclaration:
      main_function_statements.append(program_node.to_cpp())
    else:
      main_function_statements.append(program_node.to_cpp())


  for window in symbolTable.displayWindows:
      declarations.append(display_init % { 'name' : window.name,
                                           'width' : window.width,
                                           'height' : window.height,
                                           'title' : window.title })

  if len(declarations):
    app_statement = 'QApplication _app(argc, argv);'
    return_statement = 'return _app.exec();'
  else:
    app_statement = ''
    return_statement = 'return 0;'

  main_function = main_template % { 'declarations' : indent('\n'.join(declarations)),
                                    'main_code' : indent('\n'.join(main_function_statements)),
                                    'app_statement' : app_statement,
                                    'return_statement' : return_statement }

  thrust_code = preamble + '\n'.join(functions) + main_function
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

    with open(input_file, 'r') as f:
      code = ''.join(f.readlines())

    ast = parse(code)
    thrust_code = compile(ast)

    with open(output_file+'.cu', 'w') as f:
      f.write(thrust_code)


    env = { 'cuda_compiler' : cuda_compiler,
            'input_file' : output_file+'.cu',
            'output_file' : output_file }

    print('compiling...')
    pass1 = subprocess.Popen(shlex.split(cuda_compile_pass1 % env))
    pass1.wait()
    print('stage one complete')

    pass2 = subprocess.Popen(shlex.split(cuda_compile_pass2 % env))
    pass2.wait()
    print('stage two complete')

    pass3 = subprocess.Popen(shlex.split(cuda_compile_pass3 % env))
    pass3.wait()
    print('stage three complete')

    #os.remove(output_file+'.cu')
    os.remove(output_file+'.cu.o')
    os.remove(output_file+'.cu.o.NVCC-depend')


if __name__ == '__main__':
  main()


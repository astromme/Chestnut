#!/usr/bin/env python

preamble = """\
#include "libnuts/DeviceData.h"
#include "libnuts/DisplayWindow.h"
#include "libnuts/ColorKernels.h"

#include <QApplication>
"""

main_template = """\

int main(int argc, char* argv[])
{
  QApplication _app(argc, argv);
%(declarations)s
%(main_code)s

  return _app.exec();
}
"""

from parser import parse
from nodes import *


cuda_directory='/usr/local/cuda/'
cuda_compiler=cuda_directory+'bin/nvcc'

cuda_compile_pass1 = """%(cuda_compiler)s -M -D__CUDACC__ %(input_file)s -o %(input_file)s.o.NVCC-depend -m64 -Xcompiler ,\"-g\" -DNVCC -I/usr/local/cuda/include -I/usr/local/cuda/include -I/usr/include -I /Library/Frameworks/QtGui.framework/Versions/4/Headers"""

cuda_compile_pass2 = """%(cuda_compiler)s %(input_file)s -c -o %(input_file)s.o -m64 -Xcompiler ,\"-g\" -DNVCC -I/usr/local/cuda/include -I/usr/local/cuda/include -I/usr/include -I /Library/Frameworks/QtGui.framework/Versions/4/Headers"""

cuda_compile_pass3 = """/usr/bin/c++ -O2 -g -Wl,-search_paths_first -headerpad_max_install_names  ./%(input_file)s.o -o %(output_file)s /usr/local/cuda/lib/libcudart.dylib -Wl,-rpath -Wl,/usr/local/cuda/lib /usr/local/cuda/lib/libcuda.dylib -L . -lnuts -framework OpenGL -framework QtGui -framework QtOpenGL -framework QtCore"""


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

  main_function = main_template % { 'declarations' : indent('\n'.join(declarations)),
                                    'main_code' : indent('\n'.join(main_function_statements)) }

  thrust_code = preamble + '\n'.join(functions) + main_function
  return thrust_code

def main():
  import sys, os
  import shlex, subprocess

  with open(sys.argv[1], 'r') as f:
    code = ''.join(f.readlines())

  ast = parse(code)
  thrust_code = compile(ast)

  with open(sys.argv[2]+'.cu', 'w') as f:
    f.write(thrust_code)


  env = { 'cuda_compiler' : cuda_compiler,
          'input_file' : sys.argv[2]+'.cu',
          'output_file' : sys.argv[2] }

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

  #os.remove(sys.argv[2]+'.cu')
  os.remove(sys.argv[2]+'.cu.o')
  os.remove(sys.argv[2]+'.cu.o.NVCC-depend')


if __name__ == '__main__':
  main()


#!/usr/bin/env python

from parser import parse
from symboltable import Scope
from nodes import *

import pycuda.autoinit

def evaluate(ast):
  print ast

  global_env = Scope()
  return ast.evaluate(global_env)

def main():
  import sys, os
  import shlex, subprocess

  with open(sys.argv[1], 'r') as f:
    code = ''.join(f.readlines())

  ast = parse(code)
  evaluate(ast)

if __name__ == '__main__':
  main()

  # Seems to prevent crashes at the end of execution where
  # gpu objects haven't been cleaned up before autoinit finishes
  import gc
  gc.collect()



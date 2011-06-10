#!/usr/bin/env python

from parser import parse
from symboltable import SymbolTable
from nodes import *

def evaluate(ast):
  print ast

  global_env = SymbolTable()
  for program_node in ast:
    program_node.evaluate(global_env)

def main():
  import sys, os
  import shlex, subprocess

  with open(sys.argv[1], 'r') as f:
    code = ''.join(f.readlines())

  ast = parse(code)
  evaluate(ast)

if __name__ == '__main__':
  main()



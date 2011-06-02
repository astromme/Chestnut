#!/usr/bin/env python

import sys
from PyQt4 import QtGui,QtCore
from parser import parse
from symboltable import SymbolTable
from nodes import *

def evaluate(ast):
  print ast

  global_env = SymbolTable()
  for program_node in ast:
    program_node.evaluate(global_env)


def text_changed():
    print "text changed"
    #ast = parse(code)
    #evaluate(ast)

def main():
    import sys, os
    import shlex, subprocess

    with open(sys.argv[1], 'r') as f:
      code = ''.join(f.readlines())

    app = QtCore.QCoreApplication(sys.argv)

    codeWindow = QtGui.QPlainTextEdit()
    resultsWindow = QtGui.QPlainTextEdit()

    codeWindow.textChanged.connect(text_changed)

    codeWindow.show()
    resultsWindow.show()

    app.exec_()

if __name__ == '__main__':
  main()



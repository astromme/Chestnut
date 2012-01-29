#!/usr/bin/env python

import sys
from PyQt4 import QtGui,QtCore
from parser import parse
from symboltable import Scope
from nodes import *
import sys

import pycuda.autoinit

app = QtGui.QApplication(sys.argv)

window = QtGui.QWidget()
layout = QtGui.QHBoxLayout()

codeWindow = QtGui.QPlainTextEdit()
resultsWindow = QtGui.QPlainTextEdit()

class GuiStream:
  def write(self, text):
    resultsWindow.appendPlainText(text)

def evaluate(ast):
  print ast

  global_env = Scope()
  return ast.evaluate(global_env)

def text_changed():
    try:
      ast = parse(str(codeWindow.toPlainText()))
    except Exception as e:
        print e
        return

    old = sys.stdout
    resultsWindow.clear()
    sys.stdout = GuiStream()
    try:
        evaluate(ast)
    except CompilerException, e:
        print e
        return

    sys.stdout = old

def main():
    import sys, os
    import shlex, subprocess

    layout.addWidget(codeWindow)
    layout.addWidget(resultsWindow)

    window.setLayout(layout)
    window.setWindowTitle("Interpreter")

    codeWindow.textChanged.connect(text_changed)

    window.show()

    app.exec_()

if __name__ == '__main__':
  main()

  # Seems to prevent crashes at the end of execution where
  # gpu objects haven't been cleaned up before autoinit finishes
  import gc
  gc.collect()


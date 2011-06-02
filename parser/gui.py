#!/usr/bin/env python

import sys
from PyQt4 import QtGui,QtCore
from parser import parse
from symboltable import SymbolTable
from nodes import *
import sys

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

  global_env = SymbolTable()
  for program_node in ast:
    program_node.evaluate(global_env)

def text_changed():
    try:
      ast = parse(str(codeWindow.toPlainText()))
    except:
      return

    old = sys.stdout
    resultsWindow.clear()
    sys.stdout = GuiStream()
    evaluate(ast)
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



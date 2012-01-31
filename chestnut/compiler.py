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


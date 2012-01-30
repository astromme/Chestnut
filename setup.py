#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

setup(name='chestnut-compiler',
      version='0.1',
      description='Chestnut Compiler',
      author='Andrew Stromme',
      author_email='andrew.stromme@gmail.com',
      url='http://chestnutcode.org/',
      packages=['chestnut', 'chestnut.templates'],
      package_data={'chestnut' : ['templates/*.cpp']},
      #scripts = ['scripts/chestnut-compiler',
      #           'scripts/chestnut-parser'],
      long_description="""The Chestnut Compiler"""
     )

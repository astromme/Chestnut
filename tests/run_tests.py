#!/usr/bin/env python


from parser import Parser
from codeGenerator import Generator
from mipsOptimizer import MipsOptimizer
from astOptimizer import ASTOptimizer
import os,sys,signal
import subprocess


class Alarm(Exception):
  pass

def alarm_handler(signum, frame):
  frame.f_locals['self'].kill()
  raise Alarm

signal.signal(signal.SIGALRM, alarm_handler)


path = "tests"

def input_file_for(cmm_file):
    input_file = cmm_file.replace('.cmm', '.input')
    if os.path.exists(path + '/' + input_file):
        return input_file
    else:
        return None

def output_file_for(cmm_file):
    return cmm_file.replace('.cmm', '.output')

all_files = os.listdir(path)

test_cases = sorted([file for file in all_files if file.endswith("cmm") or file.endswith("c--")])
input_files = map(input_file_for, test_cases)
expected_results = map(output_file_for, test_cases)


minimal="""\
main:
_main_exit:
	jr	$ra
"""

temp = open(".minimal.mips", "w")
temp.write(minimal)
temp.close()

os.system("spim -file .minimal.mips > .minimal.output")
temp = open(".minimal.output", "r")
header = ''.join(temp.readlines())
temp.close()
os.remove(".minimal.mips")
os.remove(".minimal.output")

for test_case, input_file, expected_result in zip(test_cases,input_files, expected_results):
  print "Running %s..." % test_case, 
  sys.stdout.flush()

  temp = open("tests/" + test_case)
  if temp.readline().strip() == "//SHOULD_HANG":
    should_hang = True
  else:
    should_hang = False
  temp.close()
  
  try:
    p = Parser("tests/" + test_case)
    ast = p.parse()
  except:
    print("Parser Crashed")
    print(" " * (40 - len(test_case)) + "[\033[31mFailure\033[0m]")
    continue

  try:
    o = ASTOptimizer(ast)
    passes = o.optimizeTree()
    ast = o.ast

 
  except:
    print "AST-Optimizer Crashed",
    print(" " * (36 - len(test_case)) + "[\033[31mFailure\033[0m]")
    continue

  try:
    g = Generator(ast)
    code = g.generateLines()
  except:
    print "Generator Crashed",
    print(" " * (40 - len(test_case)) + "[\033[31mFailure\033[0m]")
    continue
  
  try:
    o = MipsOptimizer()
    code,lines_reduced,passes = o.reduce_code(code)
    if passes > 1:
      print 'MIPS optimizer removed %d lines of code' % lines_reduced
  except:
    raise
    print "MIPS-Optimizer Crashed",
    print(" " * (35 - len(test_case)) + "[\033[31mFailure\033[0m]")
    continue
  
  temp = open(".temp_code.mips", "w")
  temp.write(code.flatten())
  temp.close()


  if input_file:
    stdin = open('tests/' + input_file, 'r')
  else:
    stdin=None

  try:
    exec_process = subprocess.Popen(['spim', '-file', '.temp_code.mips'],
                    stdin=stdin, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
  except OSError:
    print "OS Error in executing process. Do you have spim installed?"
    sys.exit(1)

  signal.alarm(1) # 1 second
  try:
      (stdout, stderr) = exec_process.communicate()
  except (Alarm, KeyboardInterrupt):
    print "Program Hung",
    if should_hang:
      print(" " * (45 - len(test_case)) + "[\033[32mSuccess\033[0m]")
    else:
      print(" " * (45 - len(test_case)) + "[\033[31mFailure\033[0m]")
    continue
  finally:
    signal.alarm(0)

  result = stdout

  if stdin:
    stdin.close()

  temp = open("tests/" + expected_result)
  real_result = ''.join(temp.readlines())
  temp.close()

  result = result[len(header):]
  
  if result == real_result and should_hang == False:
    print(" " * (58 - len(test_case)) + "[\033[32mSuccess\033[0m]")
  else:
    print(" " * (58 - len(test_case)) + "[\033[31mFailure\033[0m]")
    


os.remove(".temp_code.mips")

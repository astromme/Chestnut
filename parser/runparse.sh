#!/bin/bash

if [[ "$1" == "" ]]; then
  echo "Usage: ./runparse <basefile> [infile]"
  exit 1
fi

infile=false
if [[ "$2" != "" ]]; then
  infile=true
fi

if [ $infile ]; then
  echo "==> Printing $2"
  cat $2
  echo -e "\t***"
fi

echo "==> LEX" ; lex $1.l || exit 2
echo "==> YACC" ; yacc -d $1.y || exit 3
echo "==> cc" ; cc lex.yy.c y.tab.c -o $1 || exit 4
if [ $infile ]; then
  ./$1 < $2
else
  ./$1
fi

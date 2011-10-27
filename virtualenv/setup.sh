#!/bin/sh

cd `dirname $0`

virtualenv --distribute .
pip install -E . -r ./requirements.txt $*

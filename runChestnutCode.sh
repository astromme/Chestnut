#!/bin/bash

## Moves into build directory, translates Chestnut code to Thrust code,
## compiles that Thrust code and runs it

function msg() {
  echo -e "\n\033[1m\033[1m==> $1 \033[0m"
}

#cd build

msg "Parsing Chestnut Code"
cd parser
make || exit 1
echo 
#./chestnut_parse < "chestnut.in" || exit 2
./chestnut_parse < "../DynamicChestnut.in" || exit 2

msg "Compiling Chestnut Code"
cd ../chestnut
make || exit 3

msg "Running Chestnut Code"
./chestnut_example || exit 4

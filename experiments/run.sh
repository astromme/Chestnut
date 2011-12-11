#!/bin/bash

ITERATIONS=10000
GPU=0
CPU=1

echo time ./game_of_life_cuda $ITERATIONS $CPU
echo `time ./game_of_life_cuda $ITERATIONS $CPU >/dev/null`
echo
echo time ./game_of_life_cuda $ITERATIONS $GPU
echo `time ./game_of_life_cuda $ITERATIONS $GPU >/dev/null`
echo
echo time ./game_of_life_chestnut
echo `time ./game_of_life_chestnut`
      

#!/bin/bash
ITERS=10000
DBUG=0
NUMEXPER=5


echo "@@@@@@@@@@@ Mandelbrot @@@@@@@@@@@"
echo " "
echo "Sequential:  512x512, 10000 iterations, 100 inner iters"
for ((i=0; i < $NUMEXPER; i++))
do
  time mandelbrot_sequential 
done
echo " "
echo "Chestnut: "
for ((i=0; i < $NUMEXPER; i++))
do
time mandelbrot_chestnut 
done
echo " "
echo "CUDA: "
for ((i=0; i < $NUMEXPER; i++))
do
time mandelbrot_cuda
done

echo "@@@@@@@@@@@ Game of Life @@@@@@@@@@@"
echo " "
echo "Sequential:  512x512, 10000 iterations, init'ed to zero"
for ((i=0; i < $NUMEXPER; i++))
do
time game_of_life_cuda $ITERS 1 $DBUG
done
echo " "
echo "Chestnut: "
for ((i=0; i < $NUMEXPER; i++))
do
time game_of_life_chestnut
done
echo " "
echo "CUDA: "
for ((i=0; i < $NUMEXPER; i++))
do
time game_of_life_cuda $ITERS 0 $DBUG
done

echo "@@@@@@@@@@@ Matrix Multiply @@@@@@@@@@@"
echo " "
echo "Sequential:  512x512, 100 iterations, init'ed to garbage"
for ((i=0; i < $NUMEXPER; i++))
do
time matrix_multiply_sequential
done
echo " "
echo "Chestnut: "
for ((i=0; i < $NUMEXPER; i++))
do
time matrix_multiply_chestnut
done
echo " "
echo "CUDA: "
for ((i=0; i < $NUMEXPER; i++))
do
time matrix_multiply_cuda
done

echo "@@@@@@@@@@@ Heat Flowy @@@@@@@@@@@"
echo " "
echo "Sequential:  1024x768, 10000 iterations, init with sink and source"
for ((i=0; i < $NUMEXPER; i++))
do
time heatflow_sequential
done

echo " "
echo "Chestnut: "
for ((i=0; i < $NUMEXPER; i++))
do
time heatflow_chestnut
done

echo " "
echo "CUDA: "
for ((i=0; i < $NUMEXPER; i++))
do
time heatflow_cuda
done


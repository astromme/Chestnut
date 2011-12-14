#!/bin/bash
ITERS=10000
DBUG=0
NUMEXPER=3

echo "@@@@@@@@@@@ Mandelbrot @@@@@@@@@@@"
echo " "
echo "Sequential:  512x512, 10000 iterations, 100 inner iters"
for i in {1..NUMEXPER}
do
time mandelbrot_sequential 
done
echo " "
echo "Chestnut: "
for i in {1..NUMEXPER}
do
time mandelbrot_chestnut 
done
echo " "
echo "CUDA: "
for i in {1..NUMEXPER}
do
time mandelbrot_cuda
done

echo "@@@@@@@@@@@ Game of Life @@@@@@@@@@@"
echo " "
echo "Sequential:  512x512, 10000 iterations, init'ed to zero"
for i in {1..NUMEXPER}
do
time game_of_life_cuda $ITERS 1 $DBUG
done
echo " "
echo "Chestnut: "
for i in {1..NUMEXPER}
do
time game_of_life_chestnut
done
echo " "
echo "CUDA: "
for i in {1..NUMEXPER}
do
time game_of_life_cuda $ITERS 0 $DBUG
done

echo "@@@@@@@@@@@ Matrix Multiply @@@@@@@@@@@"
echo " "
echo "Sequential:  512x512, 100 iterations, init'ed to garbage"
for i in {1..NUMEXPER}
do
time matrix_multiply_sequential
done
echo " "
echo "Chestnut: "
for i in {1..NUMEXPER}
do
time matrix_multiply_chestnut
done
echo " "
echo "CUDA: "
for i in {1..NUMEXPER}
do
time matrix_multiply_cuda
done

echo "@@@@@@@@@@@ Heat Flowy @@@@@@@@@@@"
echo " "
echo "Sequential:  1024x768, 10000 iterations, init with sink and source"
for i in {1..NUMEXPER}
do
time heatflow_sequential
done

echo " "
echo "Chestnut: "
for i in {1..NUMEXPER}
do
time heatflow_chestnut
done

echo " "
echo "CUDA: "
for i in {1..NUMEXPER}
do
time heatflow_cuda
done


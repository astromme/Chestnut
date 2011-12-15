#!/bin/bash
ITERS=100000
DBUG=0
NUMEXPER=5


echo "@@@@@@@@@@@ Mandelbrot @@@@@@@@@@@"

echo " "
echo "Chestnut:  512x512, 100000 iterations, 100 inner iters"
for ((i=0; i < $NUMEXPER; i++))
do
time bigmandelbrot_chestnut
done

echo " "
echo "CUDA: "
for ((i=0; i < $NUMEXPER; i++))
do
time bigmanelbrot
done

echo "@@@@@@@@@@@ Game of Life @@@@@@@@@@@"

echo " "
echo "Chestnut:  512x512, 100000 iterations, init'ed to zero"
for ((i=0; i < $NUMEXPER; i++))
do
time bigGOL_chestnut
done

echo " "
echo "CUDA: "
for ((i=0; i < $NUMEXPER; i++))
do
time game_of_life_cuda $ITERS 0 $DBUG
done

echo "@@@@@@@@@@@ Matrix Multiply @@@@@@@@@@@"

echo " "
echo "Chestnut:  512x512, 10000 iterations, init'ed to garbage"
for ((i=0; i < $NUMEXPER; i++))
do
time bigmatrixmult_chestnut
done

echo " "
echo "CUDA: "
for ((i=0; i < $NUMEXPER; i++))
do
time bigmatrixmult_cuda
done

echo "@@@@@@@@@@@ Heat Flowy @@@@@@@@@@@"

echo " "
echo "Chestnut:  1024x768, 100000 iterations, init with sink and source"
for ((i=0; i < $NUMEXPER; i++))
do
time bigheat_chestnut
done

echo " "
echo "CUDA: "
for ((i=0; i < $NUMEXPER; i++))
do
time bigheat_cuda
done


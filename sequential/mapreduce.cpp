/*****************************
 * Sequential Code for Benchmarking
 * ---------------------------------
 * Here we test out maps, convolutions, 
 * sorts, and reduces in a sequential 
 * environment. Change the type by
 * changing the DataType typedef cmd
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <sys/time.h>

using namespace std;

typedef float DataType;

// function declarations
void map(DataType **arr, DataType modify);
void convolve(DataType **dest, DataType **src1, DataType **src2);
void fill(DataType **arr, DataType val);
DataType reduce(DataType **arr);

DataType** createArr();
void destroyArr(DataType **arr);
void printArr(DataType **arr);

double totalTime(timeval* start, timeval* stop);

const int N = 1024;
const int rows = N;
const int cols = N;

int main(){
  
  srand (time(NULL));
  
  struct timeval start, stop;
  
  
  DataType **arr = createArr();

  fill(arr, rand()/RAND_MAX);

  gettimeofday(&start, 0);
  
  int iterations = 100;
/*
  int reduced;
  for (int i=0;i<iterations; i++){
    reduced = reduce(arr);
  }
*/

 // sorting
  vector<DataType> v(N*N);
  for (int r=0; r<rows; r++){
    for (int c=0; c<cols; c++){
      v[r*cols+c] = 0;
    }
  }
 
  for (int i=0; i<iterations; i++){
    sort(v.begin(), v.end());
  }
  
  gettimeofday(&stop, 0);
  
  cout << "time: " << totalTime(&start, &stop) << "\n";

  return 0;
}

/******************
 * Function definitions
 */
 
// reduce
DataType reduce(DataType **arr){
  DataType total = 0;
  for (int r=0; r<rows; r++){
    for (int c=0; c<cols; c++){
      total += arr[r][c];
    }
  }
  return total;
}
 
void fill(DataType **arr, DataType val){
  for (int r=0; r<rows; r++){
    for (int c=0; c<cols; c++){
      arr[r][c] = val;
    }
  }
}

// convolve
void convolve(DataType **dest, DataType **src1, DataType **src2){
  for (int r=0; r<rows; r++){
    for (int c=0; c<cols; c++){
      dest[r][c] = src1[r][c] + src2[r][c];
    }
  }
}
  
// map
void map(DataType **arr, DataType modify){
  for (int r=0; r<rows; r++){
    for (int c=0; c<cols; c++){
      arr[r][c] += modify;
    }
  }
}

DataType** createArr(){
  DataType** arr = new DataType*[rows];
  for (int r=0; r<rows; r++){
    arr[r] = new DataType[cols];
  }
  return arr;
}

void destroyArr(DataType **arr){
  for (int r=0; r<rows; r++){
    delete [] arr[r];
  }
  delete [] arr;
}

// prints out array
void printArr(DataType **arr){ 
  // print
  for (int r=0; r<rows; r++){
    for (int c=0; c<cols; c++){
      cout << arr[r][c] << " ";
    }
    cout << endl;
  }
}

double totalTime(timeval* start, timeval* stop)
{
    return (stop->tv_sec + stop->tv_usec*0.000001)
      - (start->tv_sec + start->tv_usec*0.000001);
}


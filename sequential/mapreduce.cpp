#include <iostream>
#include <algorithm>
#include <vector>
#include <sys/time.h>

using namespace std;

typedef int DataType;

void map(DataType **arr, DataType modify);
void convolve(DataType **dest, DataType **src1, DataType **src2);
void fill(DataType **arr, DataType val);
DataType reduce(DataType **arr);

DataType** createArr();
void destroyArr(DataType **arr);
void printArr(DataType **arr);

double totalTime(timeval* start, timeval* stop);


const int N = 10000;
const int rows = N;
const int cols = N;

int main(){
  
  //struct timeval start, stop;
  
  //gettimeofday(&start, 0);
  
  DataType **arr = createArr();
  //DataType **twos = createArr();

  fill(arr, 0);
  //fill(twos, 2);

  /*int iterations = 100;
  for (int i=0;i<iterations; i++){
    map(arr, 1);
  }*/
  
  //convolve(arr, arr, twos);

  //DataType reduced = reduce(arr);
  
  //vector<DataType> v (arr, arr+(rows*cols));  
  
  //gettimeofday(&stop, 0);
  
  //cout << "reduced: " << reduced << endl;
 
  //printArr(arr);
  //cout << "time: " << totalTime(&start, &stop) << "\n";

  //map(arr, 1);
  //printArr(arr);

  return 0;
}
 
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


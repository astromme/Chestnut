#include <iostream>

using namespace std;

typedef int DataType;

void map(DataType **arr, DataType modify);
void convolve(DataType **dest, DataType **src1, DataType **src2);
void fill(DataType **arr, DataType val);
DataType reduce(DataType **arr);

DataType** createArr();
void destroyArr(DataType **arr);
void printArr(DataType **arr);

const int N = 5000;
const int rows = N;
const int cols = N;

int main(){
  
  DataType **arr = createArr();
  DataType **twos = createArr();

  fill(arr, 1+2);
  fill(twos, 2);

  map(arr, 1);
  convolve(arr, arr, twos);
  map(arr, 1);

  //convolve(arr, arr, ones);
  
  DataType reduced = reduce(arr);
  cout << "reduced: " << reduced << endl;

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

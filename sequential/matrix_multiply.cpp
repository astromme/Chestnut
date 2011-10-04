int matrix_a[1000][50];
int matrix_b[50][800];
int matrix_output[1000][800];


int main() {
  for (int x=0; x<1000; x++) {
    for (int y=0; y<50; y++) {
      matrix_a[x][y] = x + y;
    }
  }

  for (int x=0; x<50; x++) {
    for (int y=0; y<800; y++) {
      matrix_b[x][y] = x + y;
    }
  }


  int iterations = 100;
  while (iterations > 0) {
    for (int x=0; x<1000; x++) {
      for (int y=0; y<800; y++) {
        matrix_output[x][y] = 0;
        for (int i=0; i<50; i++) {
          matrix_output[x][y] += matrix_a[x][i] * matrix_b[i][y]; 
        }
      }
    }
    iterations = iterations - 1;
  }

  //print(matrix_output);
}

#include <stdio.h>

#define SIZE 512
float a[SIZE][SIZE];
float b[SIZE][SIZE];
float c[SIZE][SIZE];

main() { 
  int iterations, i, j, k, result;

  iterations = 100; 
  while (iterations > 0) {
     for (i = 0; i < SIZE; i++) { 
        for (j = 0; j < SIZE; j++) {
          result = 0;
          for(k = 0; k < SIZE; k++) {
            result += a[i][k]*b[k][j];
          }
          c[i][j] = result;
        }
     }
     iterations--;
    // printf(" iteration\n");
  }

  //printf("c[0][0] = %f\n", c[0][0]);
}

//print(c);

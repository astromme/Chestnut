#include <stdio.h>
#include <stdlib.h>
#define  C0  0.6
#define  C1  0.1 // 4 elements make 0.4

#define WIDTH 1024
#define HEIGHT 768

int heat_stencil[WIDTH][HEIGHT];
float heat_data[WIDTH][HEIGHT];
float heat_copy[WIDTH][HEIGHT];

void heat_flow(float data[][HEIGHT], float copy[][HEIGHT], 
    int stencil[][HEIGHT], float c0, float c1) 
{
  for (int i=0; i < WIDTH; i++) {
    for (int j=0; j< HEIGHT; j++) {
      float new_data;
      int i_index = i-1;
      int j_index = j;
      if(i_index < 0) i_index = WIDTH-1;
      new_data = copy[i_index][j_index];  
      i_index = i+1;
      if(i_index == WIDTH) i_index = 0;
      new_data += copy[i_index][j_index];  
      j_index = j-1;
      if(j_index < 0) j_index = HEIGHT-1;
      new_data += copy[i_index][j_index];  
      j_index = j+1;
      if(j_index == HEIGHT) j_index = 0;
      new_data += copy[i_index][j_index];  

      data[i][j] = c0*data[i][j] + c1*new_data; 
      if(stencil[i][j] == 1) {
        data[i][j] = 1;
      } else if (stencil[i][j] == 2){
        data[i][j] = 0;
      }
    }
  }
}

int init(int heat_data[][HEIGHT], int heat_copy[][HEIGHT], 
    int heat_stencil[][HEIGHT]) 
{
  for (int x =0; x < WIDTH; x++) {
    for (int y =0; y < HEIGHT; y++) {
      if((x > 10 && x < 20) && (y > 50 && y < 60)) {
        heat_stencil[x][y] = 1;
      } else if((x>500 && x < 510) && (y > 400 && y < 410)) {
        heat_stencil[x][y] = 2;
      } else {
        heat_stencil[x][y] = 0;
      }
      heat_data[x][y] = 0;
      heat_copy[x][y] = 0;
    }
  }
}

int main(int argc, char *argv[]) {

  int iteration = 0;
  int max = 10000;
  if(argc > 1) {
    max = atoi(argv[1]);
  }

  while (iteration < max) {
    if(iteration %2 == 0) {
      heat_flow(heat_data, heat_copy, heat_stencil, C0, C1);
    } else {
      heat_flow(heat_copy, heat_data, heat_stencil, C0, C1);
    }
    iteration = iteration + 1;
  }
}



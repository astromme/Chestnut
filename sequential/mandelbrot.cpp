#include <cstdlib>
#include <iostream>
#include <sstream>

std::string stringFromInt(int number)
{
   std::stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}


void printFullArray2D(int* data, int width, int height) {
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      int index = width*y + x;
      std::cout << ((data[index] == 0) ? "." : stringFromInt(data[index])) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  int width = 80;
  int height = 50;

  int iterations = 1;

  int data[width*height];

  for (int it=0; it<iterations; it++) {
    for (int i=0; i<width; i++) {
      for (int j=0; j<height; j++) {
        // x0 = scaled x co-ordinate of pixel (must be scaled to lie somewhere in the interval (-2.5 to 1)
        float x0 = ((float)i)/width * 3.5 - 2.5;
        // y0 = scaled y co-ordinate of pixel (must be scaled to lie somewhere in the interval (-1, 1)
        float y0 = ((float)j)/height * 2 - 1;


        float x = 0;
        float y = 0;

        int iteration = 0;
        int max_iteration = 1000;

        while (((x*x + y*y) <= (2*2))  && ( iteration < max_iteration ))
        {
          float xtemp = x*x - y*y + x0;
          y = 2*x*y + y0;

          x = xtemp;

          iteration = iteration + 1;
        }

        int index = width*j+i;
        if ( iteration == max_iteration )
          data[index] = 0;
        else
          data[index] = iteration;
      }
    }
  }

  printFullArray2D((int*)data, width, height);
}

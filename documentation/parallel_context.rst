The Parallel Context
====================

The only way to run code in parallel is through the use of parallel contexts. These blocks of code start with a `foreach` statement and finish with the `end` keyword. A simple parallel context might look like the following ::

  foreach item in some_array
      item = 0;
  end

This snippet sets every element in `some_array` to the value 0. This code is run in parallel, think of each `item = 0` operation happening at the same time and independently from every other `item = 0` operation. The end keyword ensures that all of these operations have finished before continuing on to the next statement.

Parallel contexts can contain multiple statements and even function calls. In general anything that is a valid statement is valid inside of a parallel context. The notable exceptions are that functions marked as sequential can't be called and parallel contexts can't be nested. For example here is a parallel context that squares each element in an array using a parallel function and then multiples that result by 2 before placing it back in the array::

  parallel int square(int value) { 
      return value * value; 
  }
  
  foreach x in Xs
      x = square(x);
      x = x * 2;
  end

Multiple inputs can be used from within a parallel context. This context swaps the values in two arrays. ::

  foreach a in a_values, b in b_values
      Int temp = a;
      a = b;
      b = temp;
  end

it's easy to find the coordinates of the element that you're working with with the function `location` ::

  foreach value in some_array
      value = location(value).x;
  end

Similarly, you can get the neighbors of a value with the `window` function. This context shifts the data left by one element. ::

  foreach value in some_array
      IntWindow2d neighbors = window(value);
      value = neighbors.left;
  end

What happens beyond the edges of the array? Currently only wrap around is supported but in the future are plans for a constant value and a function-dervived value. 

External arrays can also be used from within a context::

  Array1d outputs[10];
  Array1d a_values[10];

  foreach output in outputs
      Int location = location(output).x;
      output = a_values.at(location);
  end

This makes more sense when the other arrays have different dimensions than the array that is used by the foreach. Consider this matrix multiplication example::

  RealArray2d matrix_a[1000, 500];
  RealArray2d matrix_b[500, 800];
  RealArray2d matrix_output[1000, 800];

  foreach element in matrix_output
      Int i = 0;
      element = 0;
      while (i < matrix_a.size.height) {
        element = element + matrix_a[location(element).x, i] * matrix_b[i, location(element).y]; 
        i = i + 1;
      }
  end

In this case one parallel thread is started for each element in the output matrix. That thread is responsible for figuring out how to combine the inputs to create its output. In CUDA the programmer would have to manually determine how to partition the thread space but in Chestnut it is simplified into the foreach loop.



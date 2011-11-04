The Chestnut Tutorial
====================

Ready to start learning Chestnut? This tutorial focuses on the language; if you
are interested in the graphical designer than you should look here.

Your First Program
------------------

Everybody likes hello world programs, right? I'm sorry to disappoint but we're going to start with something a little more... parallel than the traditional hello world example. Create a new file `zeros.chestnut` and put the following code in it::

  IntArray1d zeros[10];

  foreach element in zeros
      element = 0;
  end

  print(zeros);

Looks sort of C-like, right? A more traditional C program that does the same thing might look like this::

  int zeros[10];

  for (int i=0; i<10; i++) {
      zeros[i] = 0;
  }

  print(zeros);

The Python version might look like this::

  zeros = []

  for i in range(10):
      zeros.append(0)

  print(zeros)


Notice how in the C and Python versions we are specifying in what order the
array is formed? In C it is explicit, we start at i=0 and move upwards from
there. In python it is slightly more implicity but we are iterating over a list
of ten numbers [0, 1, 2, ..., 9] and running the loop for each of these
numbers. But in both cases the loop where i=3 will happen before the loop where
i=8. This might be important for some programs but almost overwhelmingly it is
not. We don't care if the zero in slot ten is written before the zero in slot
4, we just care that they are all written. Once we don't care about the order
in which loops are run it is possible to run multiple parts of the loop at
the same time. If we had a computer with 2 processors we could run half of
the computations on one and the other half on the second. Your graphics card
has dozens of cores that can run these loops at the same time; Chestnut
takes advantage of this to speed your program up. Lets say we have another
simple program::

  IntArray1d numbers[10];

  foreach number in numbers
      number = 


Printing out variables
======================

Reading and Writing Data
========================

.. function:: read(filename)

   Reads in data from the file at *filename*. Returns an array with the same
   type and dimesions as specified in the file

.. function:: write(array, filename)

   Writes the data contained in the array to *filename*. Write creates files
   which can be loaded back into chestnut using the :func:`read` function.

This example reads in a file and squares each number in the array::

  IntArray2d numbers[10, 3] = read("simple.data");

  foreach number in numbers
      number = number * number;
  end
  
  write(numbers, "simple_squared.data");

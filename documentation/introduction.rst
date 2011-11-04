An Introduction to Chestnut
===========================

Chestnut is a programming language that is focused on simple parallel
computation. Parallel computing allows a program to do many computations at the
same time to try and speed the program up. Obviously not every program can be
parallelized--if each step relies on the result of the previous step then there
are strong dependencies and each part must be run sequentially. However there
are many applications where the same operation is applied many individual
pieces of data and these computations can be run in any order. This is where
parallelism comes in.  Chestnut allows the programmer to write small contexts
of code which know about arrays of data. The code in these contexts only
operate on one element of the data at a time; Chestnut ensures that the context
runs for each element in the array. In this way parallelism is achieved because
multiple contexts can be run at the same time.

Chestnut programs are composed of a number of function declarations, statements and parallel contexts. Similar to python there is no 'main()' function that is automatically run and instead all statements in the file are run. 

The Layered Approach
--------------------

There are a few distinct parts under the Chestnut umbrella. At the top and the most removed from the low level code is the Chestnut Designer, an application for GUI programming. This designer is able to translate a graphical program into Chestnut code. The second layer is the Chestnut language and is the core of the project. This language is a high level approach to parallelism that removes the complexity of GPU and replaces it with a simple parallel model. A compiler is included which translates code written in this language into C++ source code using the Walnut library. Walnut is written in C++ and it provides abstractions to CUDA and Thrust. This library vastly simplifies the implementation of the Chestnut compiler's C++ backend because the Walnut API is much closer to the Chestnut model than the CUDA API is.

The Chestnut Designer
The Chestnut Language
The Chestnut Compiler
The Walnut Library

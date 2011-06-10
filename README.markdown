Chestnut
========

Chestnut is a language and gui for parallel programming. It produces target code that runs natively on CUDA enabled GPUs.

Dependencies
============

Chestnut depends on the following software:

* Qt Framework (version 4.5 or higher) <http://qt.nokia.com>
* Python (version 2.6 or higher) <http://python.org>
* lepl (version 5) <http://acooke.org/lepl>
* CMake (version 2.8 or higher) <http://cmake.org>
* GCC (version 4.2 or higher) <http://gcc.gnu.org>
* CUDA (version 3.2 or higher) <http://developer.nvidia.com/cuda-downloads>
* Thrust (version 1.3 or higher, included with cuda 4) <http://code.google.com/p/thrust>

Building
========

1. Download the source with git clone git://github.com/astromme/Chestnut.git
2. Create a build directory with `cd Chestnut` then  `mkdir build` then `cd build`
3. Build chestnut using cmake with `cmake ..` then `make`
4. Install chestnut with `make install`

Compiling Chestnut Code
=======================

```bash
$ chestnut-compiler file.chestnut [-o output_name]
$ ./file
```

Running the Chestnut Desginer
=============================

```bash
$ chestnut-designer
```

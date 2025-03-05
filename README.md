# Image Rotation using NVIDIA NPP with CUDA

## Overview

This project demonstrates the use of NVIDIA Performance Primitives (NPP) library with CUDA to perform image rotation. The goal is to utilize GPU acceleration to efficiently rotate a given image by a specified angle, leveraging the computational power of modern GPUs. 

## Code Organization

```bin/```
This folder contains executable code that is built automatically. 

```data/```
This folder contains example data in any format. By default it has two images.

```lib/```
Currently we don't have any thing specific. 

```src/```
Contains source code for Image Rotation.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.
This project does image rotation using NVIDIA NPP with CUDA and is hosted on GIT. GIT Repository can be cloned using following URL 

```
git clone https://github.com/gsrit31/CudaNPP
```

```INSTALL```
Contains instructions how to install this project. this is supposed to  ideally work in all operating system.

```Makefile ```
make all ---> builds the source code and creates executable in bin directory automatically.
make run ---> runs the imagerotation executable by default for Lena.png example.
if user want to run with different image input use following command
<executable_name>  --input <input_image_filename> --output <output_image_filename>

```run.sh```
alternative to running executable explicitly as shown above, user can invove with ./run.sh <arg1> <arg2>  

## Key Concepts

Uses NPP image Processing libraries and works for 2,4,8 image chanels only.

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved

## Dependencies needed to build/run
[FreeImage](../../README.md#freeimage), [NPP](../../README.md#npp)
,cuda samples. one can download and use the following commands 
cd ~
mkdir lib && cd lib
cd lib
git clone https://github.com/NVIDIA/cuda-samples.git


## Prerequisites

install the [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

- Compiling the project.
```bash
make build
```
This will build the project and creates executable under bin/ directory.

### Linux
The Linux samples are built using makefiles. 

### Windows
The source code are built on linux platform and are built using the Visual Studio IDE. It is supposed to work for windows as well.

## Running the Program
After building the project, you can run the program using the following command:

```bash
make run
```

This command will execute the compiled binary, rotating the input image (grey-sloth.png) by 90 degrees, and save the result as grey-sloth_rotated.png in the data/ directory.

If you wish to run the binary directly with custom input/output files, you can use:

```bash
./bin/imageRotationNPP --input data/grey-sloth.png --output data/grey-sloth_rotated.png
```

- Cleaning Up
To clean up the compiled binaries and other generated files, run:


```bash
make clean
```
This will remove all files in the bin/ directory.

- Sample build and execution output

make run will clean, build and run:

```bash
coder@47e8ab50863d:~/project/t2/CudaNPP$ make all
mkdir -p bin
/usr/local/cuda/bin/nvcc -std=c++17 -I/usr/local/cuda/include -Iinclude -I/home/coder/lib/cub/ -I/home/coder/lib/cuda-samples/Common src/imageRotationNPP.cpp -o bin/imageRotationNPP -L/usr/local/cuda/lib64 -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage -L/home/coder/lib
./bin/imageRotationNPP --input data/grey-sloth.png --output data/grey-sloth_rotated.png
./bin/imageRotationNPP Starting...

GPU Device 0: "Ampere" with compute capability 8.6

NPP Library Version 11.3.3
  CUDA Driver  Version: 12.6
  CUDA Runtime Version: 11.3
  Device 0: <          Ampere >, Compute SM 8.6 detected
nppiRotate opened: <data/grey-sloth.png> successfully!
Output Image sucessfully saved  : <data/grey-sloth_rotated.png
```

- Sample run with ./run.sh
```
coder@47e8ab50863d:~/project/t2/CudaNPP$ ./run.sh
No arguments provided. using default arguments to run
./bin/imageRotationNPP --input data/grey-sloth.png --output data/grey-sloth_rotated.png
./bin/imageRotationNPP Starting...

GPU Device 0: "Ampere" with compute capability 8.6

NPP Library Version 11.3.3
  CUDA Driver  Version: 12.6
  CUDA Runtime Version: 11.3
  Device 0: <          Ampere >, Compute SM 8.6 detected
nppiRotate opened: <data/grey-sloth.png> successfully!
Output Image sucessfully saved  : <data/grey-sloth_rotated.png
coder@47e8ab50863d:~/project/t2/CudaNPP$
```
- Sample run with ./run.sh and input arguments.
```
coder@47e8ab50863d:~/project/t2/CudaNPP$ ./run.sh --input data/Lena.png --output data/Lena_rotated.png
Number of arguments: 4
./bin/imageRotationNPP Starting...

GPU Device 0: "Ampere" with compute capability 8.6

NPP Library Version 11.3.3
  CUDA Driver  Version: 12.6
  CUDA Runtime Version: 11.3
  Device 0: <          Ampere >, Compute SM 8.6 detected
nppiRotate opened: <data/Lena.png> successfully!
Output Image sucessfully saved  : <data/Lena_rotated.png
coder@47e8ab50863d:~/project/t2/CudaNPP$ 
```

sample Input 
<img src="./data/grey-sloth.png">

sample Output
<img src="./data/grey-sloth_rotated.png">


|![](./data/Lena.png)|![](./data/Lena_rotated.png)|
|:-:|:-:|
|Sample Input|Sample Output after rotating 90*|



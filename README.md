# Mean Shift Image Segmentation
A Naive Implementation of MeanShift Image Segmentation

![alt text](https://randomvisionalgorithms.wordpress.com/2018/03/01/meanshift-segmentation/)

MeanShift algorithm is basically a method of finding modes in a feature space with making no assumptions about the probability distributions. For color images a typical feature space is a 5-dimensional space of three color components and two pixel coordinates.

This is a naive implementation of MeanShift image segmentation in C++ that uses OpenCV and OpenMP for some image operations and parallel computation of expensive mode calculations.

The whole algorithm implemented as a C++ class in a header file. There is an example source code that demonstrates how the class can be used for performing image segmentation (see example.cpp).

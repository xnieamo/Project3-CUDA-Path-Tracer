CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xiaomao Ding
* Tested on: Windows 8.1, i7-4700MQ @ 2.40GHz 8.00GB, GT 750M 2047MB (Personal Computer)

<img src="https://raw.githubusercontent.com/xnieamo/Project3-CUDA-Path-Tracer/master/img/cornell.2016-10-11_05-34-19z.5000samp.png" width="425"/> <img src="https://raw.githubusercontent.com/xnieamo/Project3-CUDA-Path-Tracer/master/img/cornell.2016-10-11_06-15-00z.5000samp.png" width="425"/> 

## Introduction
The code in this repository implements a CUDA-based Monte Carlo path tracer allowing us to quickly render globally illuminated images. The path tracer sends out many rays into the scene to "sample" the colors of the objects in the scene. Upon hitting an object, another ray is generated at that location. This allows the path tracer to accumulate color of light bouncing off of nearby surfaces as well. Because each new ray generated is sampled from a probability distribution described by the object's material, it takes many iterations of the algorithm before a uniform "non-noisy" image emerges.

## Features
The following features have been implemented in this project:
* Direct illumination with Multiple Importance Sampling
* Depth of field via camera jittering
* Stochastic Sample Anti-aliasing
* Realistic reflective and refractive materials via Fresnel dielectrics and conductors

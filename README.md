# Stageboard Reader

Author: [simagyari](https://github.com/simagyari)  
Version: 1.0.0

## Description
This repository contains the code and instructions for a program that reads stage data on streams based on camera images looking at stageboards. The software was created as part of the MSc dissertation **Detecting stage of upland UK streams using low-cost sensors and Computer Vision** for GEOG5285M Dissertation module of the MSc River Basin Dynamics and Management with Geographical Information Systems at the University of Leeds in the academic year 2021/2022. The code in this repository enables the users to utilise three different approaches and methods for reading stageboards based on images taken with visible light cameras. The software retrieves images from a folder and outputs the readings in a .csv file into a specified folder. More information on this in the following sections.

## Software requirements
The code was made and run using [Windows 11](https://www.microsoft.com/software-download/windows11) while Python was operated through the [Anaconda](https://www.anaconda.com/). The required packages and versions are displayed in the table below:


| **Software** | **Version** |
| :------- | :-----: |
| Anaconda | 4.13.0 |
| Python (packages below) | 3.9.12 |
| - matplotlib | 3.4.3 |
| - numpy | 1.16.6 |
| - opencv | 4.5.5 |
| - pandas | 1.2.4 |

## Installation
To install the code, download or clone the files from the [project repository](https://github.com/simagyari/stageboard-reader). The necessary files are imgrenamer.py, and one or more of the following three: stepchangereader.py, bboxreader.py, widthreader.py, depending on how many and which methods you would like to use.

The code can be installed to your preferred folder; however, it is advised to choose it so that you do not interfere with other applications.

Once installed, the software is ready to be used, provided that the [software requirements](#software-requirements) are satistifed.

## Running the application
All calibration, running, and parameter designation must happen inside an IDE or text editor that enables running and editing code, such as [Pycharm](https://www.jetbrains.com/pycharm/).
For each set of images, the software must be manually calibrated using expert judgement to create the following parameters in the following way:
1. Manually crop the image to the approximate vicinity of the stageboard, the highest part of the image should be a bit above the top of the stageboard on the image. This should yield parameters img_xmin, img_xmax, img_ymin, img_ymax.
2. Manually observe calibration image, get four points on the stageboard with known real coordinates based on the lines.
3. Construct numpy lists for orthorectification by calculating the pixel length of the stageboard from the pixel width of it (parameters ort_src, ort_dst).
4. Get pixel width and pixel height of the image from the orthorectification data, put it into the warpPerspective (parameters width, height).
5. Run orthorectification on one image to check if it is correct.
6. Check a subset of the images with the code to see if the threshold is appropriate in the thresh part. If not, correct to judgement.
7. Once satisfied with the results on the subset, run the code for the desired images specifying the folder in which they have to be run.
8. The code will output a shapefile into the set destination folder containing the characteristics of the run like location and number of images.

## What to expect when running the code.
When running the code, the end result will be a printed statement of the result location and a simple graph plotted with the time series of the derived data.

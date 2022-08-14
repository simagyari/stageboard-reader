# stageboard-reader
This repository contains the code for a Python script that reads water levels from pictures of stageboards.

Usage methodology:
1. Manually crop the image to the approximate vicinity of the stageboard, the highest part of the image should be identical to the top of the stageboard on the image.
2. Manually observe calibration image, get four points on the stageboard with known real coordinates based on the lines.
3. Construct numpy lists for orthorectification by calculating the pixel length of the stageboard from the pixel width of it.
4. Get pixel width and pixel height of the image from the orthorectification data, put it into the warpPerspective.
5. Run orthorectification on one image to check if it is correct.
6. Check a subset of the images with the code to see if the threshold is appropriate in the thresh part. If not, correct to judgement.
7. Once satisfied with the results on the subset, run the code for the desired images specifying the folder in which they have to be run.
8. The code will output a shapefile into the set destination folder containing the characteristics of the run like location and number of images.
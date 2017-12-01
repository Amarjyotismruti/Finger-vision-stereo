# Finger-vision-stereo

*The repository contains source code for generating point-clouds with two images from a stereo camera as input*  
*The output is displayed in two windows, disparity map and point cloud visualization*  
*Code is built/tested on Ubuntu 14.04 system and Mac/Windows compatibility isn't verified*  

### Required dependency libraries are: 

*[OpenCV 3.0 or above](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html)  
*[PCL 1.3 or above](http://pointclouds.org/downloads/linux.html)  


### Follow these steps for building/running the code: 

*Go to the directory ./build and run make to build the code.

*Plug in the cameras into USB sockets.

*calibrate and rectify the cmaeras and save the parameters in file "calibration.yml".(There is already a file present with my stereo sensor parameters)

*Run the executable with command ``./stereo_skin``.

*The order of images acquired from cameras in essential, so in case disparity produces garbage change the order in file "multitracker.cpp" in the code snippet:  

``    // Read the images
    VideoCapture cap2(1);
    VideoCapture cap1(0);
``



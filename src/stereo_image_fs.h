//*Fish eye lens Stereo Calibration and Rectification for Finger Vision*// 
//*12 Jul 2017*//
//*Smruti Amarjyoti ,RI, CMU*//


#ifndef stereo_image_fs
#define stereo_image_fs


#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;


namespace stereo {


void stereo_calibrate()
{

 //Create objects to access left and right camera images.
VideoCapture cap_1(0);
VideoCapture cap_2(1);

//cap_1.set(CV_CAP_PROP_FRAME_WIDTH, 320);
//cap_1.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
 if(cap_1.isOpened()==0)
 {
   cout<<"Cannot open video cam."<<endl;
   exit;
 }
 
//cap_2.set(CV_CAP_PROP_FRAME_WIDTH, 320);
//cap_2.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
 if(cap_2.isOpened()==0)
 {
   cout<<"Cannot open video cam."<<endl;
   exit;
 }
 namedWindow("Left Image",WINDOW_AUTOSIZE);
 namedWindow("Right Image",WINDOW_AUTOSIZE);
 //namedWindow("Detect",CV_WINDOW_AUTOSIZE);

 Mat image_l,image_r;
 Mat image_lg,image_rg;
 Size imageSize;
 //Acquire checkerboard size.
 const float squareSize=0.02423f;
 Size boardSize;
 cout<<"Enter the board height:"<<endl;
 cin>>boardSize.height;
 cout<<"Enter the board width:"<<endl;
 cin>>boardSize.width;
 //specify vectors for storing checkerboard image and object points.
 vector<vector<Point2f> > imagePoints[2];
 vector<vector<Point2d> > ImagePoints[2];
 vector<vector<Point3d> > objectPoints;
 //Acquire number of images to capture.
 int n_images;
 int count=0;
 cout<<"Enter the number of checkerboard image pairs to capture:"<<endl;
 cin>>n_images;
 n_images*=20;

 while(count<=n_images)
 {
    cap_1.read(image_l);
    cap_2.read(image_r);
    //converting images to grayscale.
    cvtColor(image_l,image_lg,CV_BGR2GRAY);
    cvtColor(image_r,image_rg,CV_BGR2GRAY);

    vector<Point2f> corners_1,corners_2;

    //Calculate the checkerboard corners.
    bool found_1 = findChessboardCorners(image_lg, boardSize, corners_1, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    bool found_2 = findChessboardCorners(image_rg, boardSize, corners_2, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    
    //Display corners.
    drawChessboardCorners(image_lg, boardSize, corners_1, found_1); 
    drawChessboardCorners(image_rg, boardSize, corners_2, found_2);
    //Display both the images.
    imshow("Left Image",image_lg);
    imshow("Right Image",image_rg); 
    waitKey(1);
    

    
    if( found_1 && found_2 )
    {   
        if(count!=0 && count%20==0)
      {
        //refine corners using subpixel accuracy interpolation.
        cornerSubPix(image_lg, corners_1, Size(11,11), Size(-1,-1),TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
        cornerSubPix(image_rg, corners_2, Size(11,11), Size(-1,-1),TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));

        //Save corners in vector.
        imagePoints[0].push_back(corners_1);
        imagePoints[1].push_back(corners_2);
        cout<<"Checkerboard image pair "<<count/20<<" successfully saved."<<endl;
      }
        count++;   
    }
  }
    destroyAllWindows();
    imagePoints[0].resize(n_images/20);
    imagePoints[1].resize(n_images/20);
    objectPoints.resize(n_images/20);
    imageSize=image_lg.size();

    for( int i = 0; i < n_images/20; i++ )
    {
        for( int j = 0; j < boardSize.height; ++j )
            for( int k = 0; k < boardSize.width; ++k )
                objectPoints[i].push_back(Point3d(double(k * squareSize ), double(j * squareSize ), double(0.0)));
    }


   for (int i = 0; i < imagePoints[0].size(); i++) {
    vector< Point2d > v1, v2;
    for (int j = 0; j < imagePoints[0][i].size(); j++) {
      v1.push_back(Point2d((double)imagePoints[0][i][j].x, (double)imagePoints[0][i][j].y));
      v2.push_back(Point2d((double)imagePoints[1][i][j].x, (double)imagePoints[1][i][j].y));
    }
    ImagePoints[0].push_back(v1);
    ImagePoints[1].push_back(v2);
    }

  printf("Starting Calibration\n");
  cv::Matx33d K1, K2, R;
  cv::Vec3d T;
  cv::Vec4d D1, D2;
  int flag = 0;
  flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
  flag |= cv::fisheye::CALIB_CHECK_COND;
  flag |= cv::fisheye::CALIB_FIX_SKEW;
  //flag |= cv::fisheye::CALIB_FIX_K2;
  //flag |= cv::fisheye::CALIB_FIX_K3;
  //flag |= cv::fisheye::CALIB_FIX_K4;
  cv::fisheye::stereoCalibrate(objectPoints, ImagePoints[0], ImagePoints[1],
      K1, D1, K2, D2, imageSize, R, T, flag,
      cv::TermCriteria(3, 12, 0));

  cv::FileStorage fs1("calibration.yml", FileStorage::WRITE);
  fs1 << "K1" << Mat(K1);
  fs1 << "K2" << Mat(K2);
  fs1 << "D1" << D1;
  fs1 << "D2" << D2;
  fs1 << "R" << Mat(R);
  fs1 << "T" << T;
  printf("Done Calibration\n");

  printf("Starting Rectification\n");

  cv::Mat R1, R2, P1, P2, Q;
  cv::fisheye::stereoRectify(K1, D1, K2, D2, imageSize, R, T, R1, R2, P1, P2, 
Q, CV_CALIB_ZERO_DISPARITY, imageSize, 0.0, 1.1);

  fs1 << "R1" << R1;
  fs1 << "R2" << R2;
  fs1 << "P1" << P1;
  fs1 << "P2" << P2;
  fs1 << "Q" << Q;
  fs1.release();
  printf("Done Rectification\n");

    //Produce joint undistort and rectify maps.
    Mat rmap[2][2];
    fisheye::initUndistortRectifyMap(K1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    fisheye::initUndistortRectifyMap(K2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    namedWindow("Rectified images",CV_WINDOW_AUTOSIZE);
    Mat left_rect,right_rect,rectified;
    
    while(true)
    {

      cap_1.read(image_l);
      cap_2.read(image_r);

      //converting images to grayscale.
      cvtColor(image_l,image_lg,CV_BGR2GRAY);
      cvtColor(image_r,image_rg,CV_BGR2GRAY);

      //Rectify camera images.
      remap(image_lg, left_rect, rmap[0][0], rmap[0][1], INTER_LINEAR);
      remap(image_rg, right_rect, rmap[1][0], rmap[1][1], INTER_LINEAR);

      //Blend the rectified images.
      addWeighted( left_rect, 0.3, right_rect, 0.7, 0.0, rectified);
      imshow("Rectified images",rectified);
      if (waitKey(1) == 27)
       {
            std::cout << "Esc key is pressed by user" << std::endl;
            destroyAllWindows();
            break; 
       }

    }
    //Disparity matrices.
    Mat disp,disp8;
    namedWindow("Disparity map",CV_WINDOW_AUTOSIZE);

    while(true)
   {

    cap_1.read(image_l);
    cap_2.read(image_r);

    //converting images to grayscale.
    cvtColor(image_l,image_lg,CV_BGR2GRAY);
    cvtColor(image_r,image_rg,CV_BGR2GRAY);

    //Rectify camera images.
    remap(image_lg, left_rect, rmap[0][0], rmap[0][1], INTER_LINEAR);
    remap(image_rg, right_rect, rmap[1][0], rmap[1][1], INTER_LINEAR);

    //Semi Global Disparity matching.
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,80,3);
    int sgbmWinSize = 5;
    sgbm->setBlockSize(3);
    sgbm->setPreFilterCap(4);
    sgbm->setP1(8*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(-80);
    sgbm->setNumDisparities(80);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    //sgbm->setMode(StereoSGBM::MODE_SGBM);

    sgbm->compute(left_rect, right_rect, disp);
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    imshow("Disparity map",disp8);
    if (waitKey(1) == 27)
       {
            std::cout << "Esc key is pressed by user" << std::endl;
            destroyAllWindows();
            break; 
       }
    
   }
}

void disp_map_gen( cv::Mat &image_l, cv::Mat &image_r, cv::Mat &rect_image, cv::Mat &disp_map, cv::Mat &xyz)
{
    // VideoCapture cap_1(1);
    // VideoCapture cap_2(2);
    // Mat image_l,image_r;
    Mat image_lg,image_rg;

    // cap_1.read(image_r);
    Size imageSize=image_r.size();
    cv::FileStorage fs("calibration.yml", FileStorage::READ);
    cv::Mat K1, K2;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Vec4d D1, D2;
    fs["K1"]>>K1;
    fs["K2"]>>K2;
    fs["R1"]>>R1;
    fs["R2"]>>R2;
    fs["P1"]>>P1;
    fs["P2"]>>P2;
    fs["D1"]>>D1;
    fs["D2"]>>D2;
    fs["Q"]>>Q;

    fs.release();
    //Produce joint undistort and rectify maps.
    Mat rmap[2][2];
    fisheye::initUndistortRectifyMap(K1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    fisheye::initUndistortRectifyMap(K2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    // namedWindow("Rectified images",CV_WINDOW_AUTOSIZE);
    Mat left_rect,right_rect,rectified;

    
    // while(true)
    // {

      // cap_1.read(image_l);
      // cap_2.read(image_r);

      //converting images to grayscale.
      // cvtColor(image_l,image_lg,CV_BGR2GRAY);
      // cvtColor(image_r,image_rg,CV_BGR2GRAY);

      // //Rectify camera images.
      // remap(image_lg, left_rect, rmap[0][0], rmap[0][1], INTER_LINEAR);
      // remap(image_rg, right_rect, rmap[1][0], rmap[1][1], INTER_LINEAR);

      // //Blend the rectified images.
      // addWeighted( left_rect, 0.3, right_rect, 0.7, 0.0, rectified);
      // imshow("Rectified images",rectified);
      // if (waitKey(1) == 27)
      //  {
      //       std::cout << "Esc key is pressed by user" << std::endl;
      //       destroyAllWindows();
      //       break; 
      //  }

    // }
    //Disparity matrices.
    Mat disp,disp8;

   //  while(true)
   // {

    // cap_1.read(image_l);
    // cap_2.read(image_r);

    //converting images to grayscale.
    cvtColor(image_l,image_lg,CV_BGR2GRAY);
    cvtColor(image_r,image_rg,CV_BGR2GRAY);

    //Rectify camera images.
    remap(image_lg, left_rect, rmap[0][0], rmap[0][1], INTER_LINEAR);
    remap(image_rg, right_rect, rmap[1][0], rmap[1][1], INTER_LINEAR);

    //Return the rectified image 
    rect_image=left_rect.clone();

    //Semi Global Disparity matching.
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,80,3);
    int sgbmWinSize = 5;
    sgbm->setBlockSize(5);
    sgbm->setPreFilterCap(4);
    sgbm->setP1(8*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(-120);
    sgbm->setNumDisparities(112);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    //sgbm->setMode(StereoSGBM::MODE_SGBM);

    sgbm->compute(left_rect, right_rect, disp);
    reprojectImageTo3D(disp, xyz, Q, false, CV_32F);
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    medianBlur(disp8,disp8,19);
    disp_map=disp8.clone();





}

}

#endif

// int main( int argc, char** argv )
// {


    
//     if (argv[1]=="CAL"){
//         stereo_calibrate();
//     }
//     else
//     if (argv[1]=="DISP")
//         disp_map_gen();


   // }
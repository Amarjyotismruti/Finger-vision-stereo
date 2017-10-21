#include "ros/ros.h"
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <math.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/viz/vizcore.hpp>
#include <iostream>
#include "stereo_image_fs.h"
#include "skin_viz.h"
#include "unistd.h"
using namespace std;
using namespace cv;



int main(int argc, char **argv)

{
	 // Initialize ROS
    ros::init (argc, argv, "PCL_publisher_tst");
    ros::NodeHandle nh;
    // Read the images
    VideoCapture cap2(1);
    VideoCapture cap1(0);

    Mat src_l,src_r;

    while(ros::ok())
    {
    ROS_INFO_STREAM("I am in!");
    //Test the cameras.
    cap1.read(src_l);
    // cap2.read(src_r);

    namedWindow( "IMAGE", WINDOW_AUTOSIZE );
    imshow("IMAGE",src_l);
    waitKey(0);

    }
}
//*Point cloud  generation from stereo vision*//
//18 Oct 2017*//
//*Smruti Amarjyoti, RI, CMU*//

#include <boost/thread/thread.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/median_filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>

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
#include <fstream>

using namespace std;
using namespace cv;


    // windows and trackbars name
    const std::string windowName = "Skin_features";
    const std::string ThresholdTrackbarName = "Threshold";

    // initial and max values of the parameters of interests.
    const int ThresholdInitialValue = 200;
    const int maxThreshold = 255;

    void FAST_features(const Mat& src_gray, int Threshold, const Mat& mask, std::vector<Point2f> &key_points)
    {   

        
        Mat in_img, out_img;
        // src_gray.copyTo(in_img, mask);
        // will hold the results of the detection
        // std::vector<cv::KeyPoint> key_points;
        // runs the actual detection
        // cv::FAST(in_img, key_points,80, true );
        // cv::drawKeypoints(in_img, key_points, out_img, Scalar(255,0,0));
        bool found_1 = findChessboardCorners(src_gray, Size(4,3), key_points, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        drawChessboardCorners(src_gray, Size(4,3), key_points, found_1);

        // shows the results
        imshow( windowName, src_gray);
    }

    void detect_hand(const Mat& src_gray, CascadeClassifier& hand_cascade, std::vector<cv::Rect> &key_points) {


        Point hand_centre;
        hand_cascade.detectMultiScale( src_gray, key_points, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

        for( size_t i = 0; i < key_points.size(); i++ ) {

            hand_centre.x=key_points[i].x + key_points[i].width/2;
            hand_centre.y=key_points[i].y + key_points[i].height/2;
            Point center( hand_centre.x, hand_centre.y );
            ellipse( src_gray, center, Size( key_points[i].width/2, key_points[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

          }

        imshow(windowName, src_gray);

    }

boost::shared_ptr<pcl::visualization::PCLVisualizer> stereo_vis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  //Create the pcl viewer object
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Stereo Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}



int main(int argc, char** argv)
{
    //Manage commandline arguments.
    bool store = false;
    if (argc>1)
    {
        std::string arg = argv[1];
    
        if (arg=="--save_z")
            store=true;
    }
       
    //Declare the point cloud variable
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Stereo Viewer"));
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new   pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_filtered(new   pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_filtered2(new   pcl::PointCloud<pcl::PointXYZ>);
    pointcloud->header.frame_id = "tf_frame";
    pointcloud->width = 1;
    pointcloud->height = 1;
    pcl::PointXYZ point1(1,1,1);
    pointcloud->points.push_back(point1);
    viewer = stereo_vis(pointcloud);
    //Initialize cascade object for hand detection
    // String hand_cascade_name = "haarcascades/fist.xml";
    // CascadeClassifier hand_cascade;
    // hand_cascade.load(hand_cascade_name);
    int iter = 0;
    std::vector<int> index;
    // Create the filtering objects.
    pcl::VoxelGrid<pcl::PointXYZ> vxg;
    vxg.setLeafSize (0.01f, 0.01f, 0.01f);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setRadiusSearch(0.01);
    outrem.setMinNeighborsInRadius (5);
    sor.setMeanK (200);
    sor.setStddevMulThresh (0.07);
    pcl::PassThrough<pcl::PointXYZ> vox;
    vox.setFilterFieldName ("z");
    vox.setFilterLimits (0.1, 0.35);    //camera-1-props 0.07-0.2/15
    // vox.setFilterLimits (0.1,0.3);   //camera-2-props
    // pcl::MedianFilter<pcl::PointXYZ> med_f;
    // med_f.setWindowSize(10);
    // med_f.setMaxAllowedMovement(15.0);


    Mat src_l, src_r ,src_gray;
    Mat img_rect, img_disp, nd_points;
    std::vector<cv::Point2f> key_points;
    std::vector<cv::Rect> hand_rect;
    // Read the images
    VideoCapture cap2(1);
    VideoCapture cap1(0);
    
    //Test the cameras.
    cap1.read(src_l);
    cap2.read(src_r);
    waitKey(25);
    // usleep(1000000);
    if( src_l.empty() && src_r.empty() )
    {
        std::cerr<<"Invalid input image\n";
        return -1;
    }

    //Initialize file to save z values.
    
    ofstream fout("depth_values.txt");
    
    // infinite loop to display.
    // and refresh the content of the pointcloud.
    char key = 0;

    while(!viewer->wasStopped())
    { 
      cap1.read(src_l);
      cap2.read(src_r);
      
      if( src_l.empty() && src_r.empty() )
     {
        std::cerr<<"Invalid input image\n";
        return -1;
     }

      //Enhance the contrast and brightness of the images.
      src_r.convertTo(src_r, -1, 2, 30);
      src_l.convertTo(src_l, -1, 2, 30);


      // Reduce the noise so we avoid false circle detection
      // GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
        stereo::disp_map_gen(src_r, src_l, img_rect, img_disp, nd_points);
        // threshold( img_disp, img_disp, 133, 255, 4 );
        imshow("Disparity map",img_disp);
        normalize(img_disp, img_disp, 0, 255, CV_MINMAX, CV_8U);
        // runs the detection, and update the display
        // FAST_features(img_rect, Threshold, mask, key_points);
        //Detet hand in the image and return centre.
        // detect_hand(img_rect, hand_cascade, hand_rect);
        pointcloud->clear();
        // pointcloud_filtered->clear();
        //Store the depth points in point cloud.
        //Strccture the point cloud data structure.
        //pointcloud->resize(img_disp.rows*img_disp.cols);
        pointcloud->width = (img_rect.cols);
        pointcloud->height =(img_rect.rows);
        pointcloud->is_dense = true;
        pcl::PointXYZ point;

        //Construct the point cloud from the XYZ image from OpenCV Mat data.
        for (int i(0); i < img_disp.rows; i++)
        {
            for (int j(0); j < img_disp.cols; j++)
            {   
                if (img_disp.at<uchar>(i,j)==0)
                    continue;
                Point3f p = nd_points.at<Point3f>(i, j);
                point.x = p.x*25;
                point.y = p.y*25;
                point.z = p.z*25;  //Change to -15 for camera 2.

                if (store)
                {

                    fout<<point.z<<endl;
                }
                //std::cout<<"Z value:"<<point.z<<std::endl;
                pointcloud->points.push_back(point);


            }
        }

        vxg.setInputCloud(pointcloud);
        vxg.filter(*pointcloud_filtered);
        //remove noise from the point cloud.
        vox.setInputCloud(pointcloud_filtered);
        // std::cout<<"I am in!"<<std::endl;
        vox.filter(*pointcloud);
        sor.setInputCloud(pointcloud);
        sor.filter(*pointcloud_filtered);
        // pcl::removeNaNFromPointCloud(*pointcloud, *pointcloud_filtered, index);
        // std::cout<<"Updated Viewer"<<std::endl;
        viewer->updatePointCloud(pointcloud_filtered, "sample cloud");
        key = (char)waitKey(15);
        viewer->spinOnce();

    }

    return 0;

}
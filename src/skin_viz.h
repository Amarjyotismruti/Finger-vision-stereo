#ifndef skin_viz
#define skin_viz

#include <opencv2/viz/vizcore.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

class viz_skin {

/// Create a window

// viz::Viz3d myWindow("Coordinate Frame");
// std::vector<viz::WSphere> sphere_widget(50);

public:

void display_markers(const vector<KeyPoint> &keys, cv::Mat &disp);
void init();


};

void viz_skin::init(){

    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");

    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    for (int i=0; i<50; i++ ){

    char name[8];
    sprintf(name,"sphere%d",i);
    /// Construct a sphere widget
    viz::WSphere sphere_widget(Point3f(i*0.5,0.5,0.0),(double)0.02, 10, viz::Color::white());
    // // cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    // /// Display widget (update if already displayed)
    cout<<name;
    myWindow.showWidget(name, sphere_widget);
    }



}

void viz_skin::display_markers(const vector<KeyPoint> &keys, cv::Mat &disp) {
// int main() {

    viz::Viz3d myWindow = viz::getWindowByName("Coordinate Frame");
    // Rodrigues vector
    Mat rot_vec = Mat::zeros(1,3,CV_32F);
    float translation = 0.0,x,y,z;

    // while(!myWindow.wasStopped())
    // {
    	for (int i=0; i<50 ;++i) {
		    char name[8];
    		sprintf(name,"sphere%d",i);
    		x=keys[i].pt.x/255.0;
    		y=keys[i].pt.y/255.0;
    		z=disp.at<float>(x,y)/200;
    		Affine3f pose(rot_vec, Vec3f(x, y, z));
	        myWindow.setWidgetPose(name, pose);


    	}

        myWindow.spinOnce(1, true);
    // }



}




#endif
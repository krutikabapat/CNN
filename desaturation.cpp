// Desaturation of images
// Make it more sharp or coloras added

#include<iostream>
#include<opencv2/imgproc.hpp>
#include<vector>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;


int main(){
	
	Mat image = imread("/home/krutika/Documents/Summer_Projects/Computer_Vision_Course/rgb.jpg",IMREAD_COLOR);
	int scale = 2;

	Mat hsvImage;

	cv::cvtColor(image,hsvImage,COLOR_BGR2HSV);

	hsvImage.convertTo(hsvImage,CV_32F);

	vector<Mat>channel(3);
	split(hsvImage,channel);

	channel[1] = channel[1]*scale;
	min(channel[1],255,channel[1]);
  	max(channel[1],0,channel[1]);


  	 Mat imSat;
  // Convert to BGR color space
  cv::cvtColor(hsvImage,imSat,COLOR_HSV2BGR);

  // Display the images
  Mat combined;
  cv::hconcat(image, imSat, combined);
  namedWindow("Original Image   --   Desaturated Image",CV_WINDOW_AUTOSIZE);

  imshow("Original Image   --   Desaturated Image",combined); 
  cv::imwrite("results/desaturated.jpg",imSat);

  // Wait for user to press a key
  waitKey(0);
  destroyAllWindows();

  return 0;
}



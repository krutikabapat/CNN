#include<vector>
#include<iostream>
#include<opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

//Io = I+Beta
//where B is the brightness factor

int main()
{
	Mat image = imread("/home/krutika/Documents/Summer_Projects/Computer_Vision_Course/rgb.jpg",IMREAD_COLOR);

	int Beta = 100;
	Mat ycb;
	cv::cvtColor(image,ycb,COLOR_BGR2YCrCb);

	ycb.convertTo(ycb,CV_32F);

	vector<Mat>channels(3);
	split(ycb,channels);

	channels[0]= channels[0]+Beta;
	min(channels[0],255,channels[0]);
	 min(channels[0],255,channels[0]);
  max(channels[0],0,channels[0]);  
  
  // Merge the channels 
  merge(channels,ycb);
  
  // Convert back from float32
  ycb.convertTo(ycb,CV_8UC3);

  Mat brightImage;

  // Convert back to BGR
  cv::cvtColor(ycb,brightImage,COLOR_YCrCb2BGR);

  // Display and save the images
  Mat combined;
  cv::hconcat(image, brightImage, combined);
  namedWindow("Original Image   --   Brightness Enhancement",CV_WINDOW_AUTOSIZE);
  cv::imshow("Original Image   --   Brightness Enhancement",combined);
  cv::imwrite("results/bright.jpg",brightImage);

  // Wait for user to press a key
  waitKey(0);
  destroyAllWindows();

  return 0;
}


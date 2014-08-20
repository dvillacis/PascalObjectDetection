#include <boost/foreach.hpp>
#include "ObjectDetector.h"

#define WIN_SIZE_NMS_KEY   "nms_win_size"
#define RESP_THESH_KEY     "sv_response_threshold"
#define OVERLAP_THRESH_KEY "detection_overlap_threshold"

using namespace cv;
using namespace std;

// Object Detector class

ObjectDetector::ObjectDetector(vector<float> svmDetector):
    _svmDetector(svmDetector)
{
    _hog.setSVMDetector(_svmDetector);
}

ObjectDetector::~ObjectDetector()
{
}

void ObjectDetector::getDetections(Mat img, vector<Rect>& found)
{
    Size winStride = Size(8,8);
    Size padding = Size(32,32);
    double scale0 = 1.05;
    double finalThreshold = 0.0;

    Mat grayImg;
    cv::cvtColor(img, grayImg, CV_RGB2GRAY);

    _hog.detectMultiScale(grayImg,found,0,winStride,padding,scale0,finalThreshold,true);

    cout << "Getting detections "<< found.size() <<" !!" << endl;
}






















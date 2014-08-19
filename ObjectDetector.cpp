#include <boost/foreach.hpp>
#include "ObjectDetector.h"

#define WIN_SIZE_NMS_KEY   "nms_win_size"
#define RESP_THESH_KEY     "sv_response_threshold"
#define OVERLAP_THRESH_KEY "detection_overlap_threshold"

using namespace cv;
using namespace std;

// Object Detector class

ObjectDetector::ObjectDetector(FeatureExtractor* featExtractor, SupportVectorMachine svm):
    _featExtractor(featExtractor),
    _svm(svm)
{
}

ObjectDetector::~ObjectDetector()
{
}

void ObjectDetector::getDetections(Mat img)//, 
        //FeatureExtractor* featExtractor, SupportVectorMachine svm)//, 
        //vector<Detection>& dets)
{
    Size winStride = Size(8,8);
    Size padding = Size(32,32);
    double scale0 = 1.05;
    double finalThreshold = 2.0;

    Rect o(0,0,img.cols,img.rows);
    vector<Rect> allCandidates;

    for(int j = 0; j < img.rows; j = j + 128)
    {
        for(int i = 0; i < img.cols; i = i + 64)
        {
            Rect r(i,j,i+64,j+128);
            Rect intersection = r & o;
            Mat patch = img(intersection);
            resize(patch,patch,Size(64,128));

            Feature f;
            (*_featExtractor)(patch,f);
            _svm.predict(f);
            // if(_svm.predictLabel(f) > 0){
            //     // Possible detection
            //     LOG(INFO) << "Found possible detection: " << intersection;
            //     allCandidates.push_back(intersection);
            // }

    //         //patch.release();
        }
    }

    // cout << allCandidates.size() << endl;
}






















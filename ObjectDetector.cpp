#include <boost/foreach.hpp>
//#include <modules/objdetect/src/hog.cpp>
#include "ObjectDetector.h"

#define WIN_SIZE_NMS_KEY   "nms_win_size"
#define RESP_THESH_KEY     "sv_response_threshold"
#define OVERLAP_THRESH_KEY "detection_overlap_threshold"

using namespace cv;
using namespace std;

// Object Detector class

ObjectDetector::ObjectDetector(const SupportVectorMachine& svm):
    _svm(svm)
{
    _svmDetector = svm.getDetector();
}

ObjectDetector::~ObjectDetector()
{
    //HOGDescriptor::~HOGDescriptor();
}

void ObjectDetector::getDetections(Mat img, vector<Rect>& found)
{
    //TODO: Put the hit theshold to be configurable from the outside
    _winSize = Size(64,128);
    _blockSize = Size(16,16);
    _blockStride = Size(8,8);
    _cellSize = Size(8,8);
    _nbins = 9;

    HOGDescriptor hog(_winSize,_blockSize,_blockStride,_cellSize,_nbins);
    hog.setSVMDetector(_svmDetector);

    vector<Point> hits;
    vector<Point> locations;
    vector<double> weights;

    // Mat grayImg;
    // cv::cvtColor(img, grayImg, CV_RGB2GRAY);
    // Mat testImg;
    // blur(img, testImg, Size(3,3));
    // Canny(testImg, testImg, 2, 2*3, 3);

    // Detecting on first level of the pyramid
    // Mat imgDown;
    // pyrDown(img,imgDown,Size(img.cols/2,img.rows/2));
    detect(img,hits,weights,1,Size(16,16),Size(0,0),locations, &hog);
    //HOGDescriptor::detect(img,hits,weights,0.0,Size(8,8),Size(32,32),locations);
    for(int i = 0; i < hits.size(); i++)
    {
        Rect r(hits[i],Size(64,128));
        //Rect r(Point(hits[i].x*(2),hits[i].y*(2)),Size(64*(2),128*(2)));
        found.push_back(r);
        cout << r << " score: " << weights[i] << endl;
    }

    // Detecting on one scale down
    // int num_levels = 2;
    // Mat imgDown = img;
    // for(int i = 1; i <= num_levels; ++i){
        // cout << i << endl;
    // double scale = 1;

    // Mat imgDown;
    // int i = 1;
    // hits.clear();
    // locations.clear();
    // weights.clear();
    // pyrDown(img,imgDown,Size(img.cols/scale,img.rows/scale));
    // detectLinearKernel(imgDown,hits,weights,4,Size(8,8),Size(32,32),locations);
    // for(int j = 0; j < hits.size(); j++)
    // {
    //     Rect r(Point(hits[j].x*(scale*i),hits[j].y*(scale*i)),Size(64*(scale*i),128*(scale*i)));
    //     found.push_back(r);
    //     cout << r << " score: " << weights[j] << endl;
    // }

    // Mat imgUp;
    // hits.clear();
    // locations.clear();
    // weights.clear();
    // pyrUp(img,imgUp,Size(img.cols*scale,img.rows*scale));
    // detectLinearKernel(imgUp,hits,weights,4,Size(8,8),Size(32,32),locations);
    // for(int j = 0; j < hits.size(); j++)
    // {
    //     Rect r(Point(hits[j].x/(scale*i),hits[j].y/(scale*i)),Size(64/(scale*i),128/(scale*i)));
    //     found.push_back(r);
    //     cout << r << " score: " << weights[j] << endl;
    // }
    //}


    //HOGDescriptor::detectMultiScale(img,found, 0, Size(8,8), Size(32,32), 1.05, 3,true);
}

void ObjectDetector::detect(const Mat& img, vector<Point>& hits, vector<double>& weights, 
        double hitThreshold, Size winStride, Size padding, const vector<Point>& locations, HOGDescriptor* hog)
{
    for(int i = 0; i < img.cols-_winSize.width; i=i+winStride.width)
    {
        for(int j = 0; j < img.rows-_winSize.height; j=j+winStride.height)
        {
            Rect r(Point(i,j),_winSize);
            Mat patch = img(r);
            vector<float> patchWeights;
            hog->compute(patch, patchWeights, Size(8,8), Size(0,0));
            double score;
            float predictedLabel = _svm.predictLabel(patchWeights,score);
            if(predictedLabel > 0)
            {
                hits.push_back(Point(i,j));
                weights.push_back(score);
            }
        }
    }
    
}
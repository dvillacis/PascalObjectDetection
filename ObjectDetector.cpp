#include <boost/foreach.hpp>
#include <modules/objdetect/src/hog.cpp>
#include "ObjectDetector.h"

#define WIN_SIZE_NMS_KEY   "nms_win_size"
#define RESP_THESH_KEY     "sv_response_threshold"
#define OVERLAP_THRESH_KEY "detection_overlap_threshold"

using namespace cv;
using namespace std;

// Object Detector class

ObjectDetector::ObjectDetector()
{
    HOGDescriptor::HOGDescriptor();
}

ObjectDetector::~ObjectDetector()
{
    HOGDescriptor::~HOGDescriptor();
}

void ObjectDetector::getDetections(Mat img, vector<Rect>& found)
{
    vector<Point> hits;
    vector<Point> locations;
    vector<double> weights;

    //detectLinearKernel(img,hits,weights,2.0,Size(8,8),Size(32,32),locations);
    HOGDescriptor::detect(img,hits,weights,0.0,Size(8,8),Size(32,32),locations);
    for(int i = 0; i < hits.size(); i++)
    {
        cout << hits[i] << endl;
        Rect r(hits[i],Size(64,128));
        found.push_back(r);
    }
    //HOGDescriptor::detectMultiScale(img,found, 0, Size(8,8), Size(32,32), 1.05, 3,true);
}

void ObjectDetector::detectLinearKernel(const Mat& img, vector<Point>& hits, vector<double>& weights, 
        double hitThreshold, Size winStride, Size padding, const vector<Point>& locations)
{
    hits.clear();
    if(svmDetector.empty())
        throw "No SVM Detector defined";

    Size cacheStride(gcd(winStride.width, blockStride.width),gcd(winStride.height,blockStride.height));
    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();

    double rho = svmDetector.size() > dsize ? svmDetector[dsize] : 0;
    vector<float> blockHist(blockHistogramSize);

    for( size_t i = 0; i < nwindows; i++ )
    {
        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }
        double s = rho;
        const float* svmVec = &svmDetector[0];
        int j, k;
        for( j = 0; j < nblocks; j++, svmVec += blockHistogramSize )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;
            const float* vec = cache.getBlock(pt, &blockHist[0]);
            for( k = 0; k <= blockHistogramSize - 4; k += 4 )
                s += vec[k]*svmVec[k] + vec[k+1]*svmVec[k+1] + vec[k+2]*svmVec[k+2] + vec[k+3]*svmVec[k+3];
            for( ; k < blockHistogramSize; k++ )
                s += vec[k]*svmVec[k];
        }
        if( s >= hitThreshold )
        {
            hits.push_back(pt0);
            weights.push_back(s);
        }
    }
    
}

void ObjectDetector::detectRBFKernel(const Mat& img, vector<Point>& hits, vector<double>& weights, 
        double hitThreshold, Size winStride, Size padding, const vector<Point>& locations)
{
    hits.clear();
    if(svmDetector.empty())
        throw "No SVM Detector defined";

    Size cacheStride(gcd(winStride.width, blockStride.width),gcd(winStride.height,blockStride.height));
    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();

    double rho = svmDetector.size() > dsize ? svmDetector[dsize] : 0;
    vector<float> blockHist(blockHistogramSize);

    for( size_t i = 0; i < nwindows; i++ )
    {
        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }
        double s = rho;
        const float* svmVec = &svmDetector[0];
        int j, k;
        for( j = 0; j < nblocks; j++, svmVec += blockHistogramSize )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;
            const float* vec = cache.getBlock(pt, &blockHist[0]);
            for( k = 0; k <= blockHistogramSize - 4; k += 4 )
                s += vec[k]*svmVec[k] + vec[k+1]*svmVec[k+1] + vec[k+2]*svmVec[k+2] + vec[k+3]*svmVec[k+3];
            for( ; k < blockHistogramSize; k++ )
                s += vec[k]*svmVec[k];
        }
        if( s >= hitThreshold )
        {
            hits.push_back(pt0);
            weights.push_back(s);
        }
    }
    
}

void ObjectDetector::setDetector(vector<float> svmDetector)
{
    HOGDescriptor::setSVMDetector(svmDetector);
    LOG(INFO) << "SVMDetector initialized";
}






















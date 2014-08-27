#include <boost/foreach.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>

#include "ObjectDetector.h"

#define WIN_SIZE_NMS_KEY   "nms_win_size"
#define RESP_THESH_KEY     "sv_response_threshold"
#define OVERLAP_THRESH_KEY "detection_overlap_threshold"

using namespace boost::accumulators;
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

void ObjectDetector::getDetections(Mat img, vector<Detection>& found)
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

    float hitThreshold = -1;


    // vector<Rect> f;
    // vector<double> w;
    // hog.detectMultiScale(img, f, w, hitThreshold, Size(32,32), Size(0,0), 1.05,6);
    // for(int i = 0; i < f.size(); i++)
    // {
    //     Detection det(f[i],w[i]);
    //     found.push_back(det);
    //     cout << det << endl;
    //     // if(w[i] > hitThreshold)
    //     // {
    //     //     Detection det(f[i],w[i]);
    //     //     found.push_back(det);
    //     //     cout << det << endl;
    //     // }
    // }

    detect(img,hits,weights,hitThreshold,Size(16,16),Size(0,0),locations, &hog);
    //HOGDescriptor::detect(img,hits,weights,0.0,Size(8,8),Size(32,32),locations);
    for(int i = 0; i < hits.size(); i++)
    {
        Rect r(hits[i],Size(64,128));
        Detection det(r,weights[i]);
        found.push_back(det);
        cout << det << endl;
    }

    // // cout << found.size() << endl;
    // // groupRectangles(found, 2, 0.2);
    // // cout << found.size() << endl;

    cout << "Detecting on upper pyramid" << endl;
    Mat imgDown;
    hits.clear();
    locations.clear();
    weights.clear();
    pyrDown(img,imgDown,Size(img.cols/2,img.rows/2));
    detect(imgDown,hits,weights,hitThreshold,Size(8,8),Size(0,0),locations, &hog);
    for(int j = 0; j < hits.size(); j++)
    {
        Rect r(Point(hits[j].x*2,hits[j].y*2),Size(64*2,128*2));
        Detection det(r,weights[j]);
        found.push_back(det);
        cout << det << endl;
    }

    // cout << "Detecting on lower pyramid" << endl;
    // Mat imgUp;
    // hits.clear();
    // locations.clear();
    // weights.clear();
    // pyrUp(img,imgUp,Size(img.cols*2,img.rows*2));
    // detect(imgUp,hits,weights,hitThreshold,Size(32,32),Size(0,0),locations, &hog);
    // for(int j = 0; j < hits.size(); j++)
    // {
    //     Rect r(Point(hits[j].x/2,hits[j].y/2),Size(64/2,128/2));
    //     found.push_back(r);
    //     cout << r << " score: " << weights[j] << endl;
    // }
    
    //groupRectangles(found,weights,4,0.2);

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

            vector<float> features;
            int num_features = patchWeights.size();
            accumulator_set<float, stats<tag::max, tag::min> > acc;
            for(int k = 0; k < num_features; ++k)
                acc(patchWeights[k]);
            //float mu = boost::accumulators::mean(acc);
            //float std = sqrt(moment<2>(acc));
            float xmax = boost::accumulators::max(acc);
            float xmin = boost::accumulators::min(acc);

            for(int i = 0; i < num_features; i++)
            {
                //features.push_back((patchWeights[i]-mu)/std);
                features.push_back((patchWeights[i]-xmin)/(xmax-xmin));
                //cout << features[i] << endl;
            }

            double score;
            float predictedLabel = _svm.predictLabel(features,score);
            if(predictedLabel > 0) //&& score > hitThreshold)
            {
                hits.push_back(Point(i,j));
                weights.push_back(score);
            }
        }
    }
    
}

void ObjectDetector::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps)
{
    cout << "Grouping rectangles" << endl;
    if( groupThreshold <= 0 || rectList.empty() )
    {
        return;
    }

    CV_Assert(rectList.size() == weights.size());

    vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));
    cout << "nclasses: " << nclasses << endl;

    vector<cv::Rect_<double> > rrects(nclasses);
    vector<int> numInClass(nclasses, 0);
    vector<double> foundWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();

    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        foundWeights[cls] = cv::max(foundWeights[cls], weights[i]);
        numInClass[cls]++;
    }

    for( i = 0; i < nclasses; i++ )
    {
        // find the average of all ROI in the cluster
        cv::Rect_<double> r = rrects[i];
        double s = 1.0/numInClass[i];
        rrects[i] = cv::Rect_<double>(cv::saturate_cast<double>(r.x*s),
            cv::saturate_cast<double>(r.y*s),
            cv::saturate_cast<double>(r.width*s),
            cv::saturate_cast<double>(r.height*s));
    }

    rectList.clear();
    weights.clear();

    for( i = 0; i < nclasses; i++ )
    {
        cv::Rect r1 = rrects[i];
        int n1 = numInClass[i];
        double w1 = foundWeights[i];
        cout << "n1: " << n1 << " groupThreshold: " << groupThreshold << endl;
        // if( n1 <= groupThreshold )
        //     continue;
        // filter out small rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = numInClass[j];

            if( j == i || n2 <= groupThreshold )
                continue;

            cv::Rect r2 = rrects[j];

            int dx = cv::saturate_cast<int>( r2.width * eps );
            int dy = cv::saturate_cast<int>( r2.height * eps );

            if( r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            cout << "Adding r1: " << r1 << " weight: " << w1 << endl;
            rectList.push_back(r1);
            weights.push_back(w1);
        }
    }
}
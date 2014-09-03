#include "Feature.h"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>

using namespace boost::accumulators;
using namespace cv;

void FeatureExtractor::operator()(Mat &img, Feature &feat) const
{
    this->operator()(img, feat);
}

void FeatureExtractor::operator()(const PascalImageDatabase &db, FeatureCollection &feats) const
{
    int n = db.getSize();

    feats.resize(n);
    float percent;
    for(int i = 0; i < n; i++) {
        printf("\033[s");
        // Print progress string
        if((i+1)%1000 == 0 || (i+1) == n)
        {
            percent = ((i+1)*100)/n;
            cout << percent << "% ... ";
            fflush(stdout);
            printf("\033[u");
        }

        Mat img = imread(db.getFilename(i).c_str());
        Rect roi = db.getRoi(i);
        bool flipped = db.isFlipped(i);
        Mat patch = img(roi);
        if(flipped == true)
            flip(patch, patch,1);

        // if(db.getLabel(i)>0){
        //     rectangle(img,roi,Scalar(255,0,0),2);
        //     imshow("Patch", img);
        //     cout << "************" << endl;
        //     cout << db.getLabel(i) << " - " << flipped << endl;
        //     waitKey(0);
        // }

        (*this)(patch, feats[i]);
    }
    cout << endl;
}

FeatureExtractor * FeatureExtractor::create(const std::string &featureType, const ParametersMap &params)
{
    ParametersMap tmp = params;
    tmp[FEATURE_TYPE_KEY] = featureType;
    return FeatureExtractor::create(tmp);
}

FeatureExtractor * FeatureExtractor::create(ParametersMap params)
{
    std::string featureType = params.getStr(FEATURE_TYPE_KEY);
    params.erase(FEATURE_TYPE_KEY);

    if(strcasecmp(featureType.c_str(), "hog") == 0) 
        return new HOGFeatureExtractor(params);
    else {
        throw std::runtime_error("ERROR: Unknown feature type: " + featureType);
    }
}

ParametersMap FeatureExtractor::getDefaultParameters(const std::string &featureType)
{
    ParametersMap params;
    if(strcasecmp(featureType.c_str(), "hog"     ) == 0) 
        params = HOGFeatureExtractor::getDefaultParameters();
    else {
        throw std::runtime_error("ERROR: Unknown feature type: " + featureType);
    }

    params[FEATURE_TYPE_KEY] = featureType;

    return params;
}

void FeatureExtractor::save(FILE *f, const FeatureExtractor *feat)
{
    ParametersMap params = feat->getParameters();
    params[FEATURE_TYPE_KEY] = feat->getFeatureType();
    params.save(f);
}

FeatureExtractor * FeatureExtractor::load(FILE *f)
{
    ParametersMap params;
    params.load(f);
    return FeatureExtractor::create(params);
}

void FeatureExtractor::scale(FeatureCollection &featureCollection,  FeatureCollection &scaledFeatureCollection)
{
    vector<float> feature_max(featureCollection[0].size(),0.0);
    vector<float> feature_min(featureCollection[0].size(),0.0);

    // Fill the max and min vectors
    for(int i = 0; i < featureCollection.size(); i++)
    {
        Feature f = featureCollection[i];
        for(int j = 0; j < f.size(); j++)
        {
            feature_max[j] = std::max(feature_max[j],f[j]);
            feature_min[j] = std::min(feature_min[j],f[j]);
        }
    }

    // Scale the feature collection
    for(int i = 0; i < featureCollection.size(); i++)
    {
        Feature f = featureCollection[i];
        Feature scaledF;
        for(int j = 0; j < feature_max.size(); j++)
        {
            float value = f[j];
            if(value == feature_min[j])
                value = -1;
            else if(value == feature_max[j])
                value = 1;
            else
            {
                value = -1 + (2 * ((f[j]-feature_min[j])/(feature_max[j]-feature_min[j])));
            }      
            scaledF.push_back(value);          
        }

        scaledFeatureCollection.push_back(scaledF);

    }
}

// ============================================================================
// HOG
// ============================================================================

const char *N_ANGULAR_BINS_KEY     = "n_angular_bins";
const char *UNSIGNED_GRADIENTS_KEY = "unsigned_gradients";
const char *CELL_SIZE_KEY          = "cell_size";

ParametersMap HOGFeatureExtractor::getDefaultParameters()
{
    ParametersMap params;
    params.set(N_ANGULAR_BINS_KEY    , 9);
    params.set(UNSIGNED_GRADIENTS_KEY, 1);
    params.set(CELL_SIZE_KEY         , 16);
    return params;
}

ParametersMap HOGFeatureExtractor::getParameters() const
{
    ParametersMap params;
    params.set(N_ANGULAR_BINS_KEY    , _nAngularBins);
    params.set(UNSIGNED_GRADIENTS_KEY, _unsignedGradients);
    params.set(CELL_SIZE_KEY         , _cellSize);
    return params;
}

HOGFeatureExtractor::HOGFeatureExtractor(const ParametersMap &params)
{
    _nAngularBins = params.getInt(N_ANGULAR_BINS_KEY);
    _unsignedGradients = params.getInt(UNSIGNED_GRADIENTS_KEY);
    _cellSize = params.getInt(CELL_SIZE_KEY);

}

void HOGFeatureExtractor::operator()(Mat &img, Feature &feat) const
{
    //Converting the image to grayscale
    // Mat grayImg;
    // cv::cvtColor(img, grayImg, CV_RGB2GRAY);

    //cout << grayImg.size() <<endl;
    // blur(img, img, Size(3,3));
    // Canny(img, img, 2, 2*3, 3);
    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);

    resize(img,img,hog.winSize);

    // if(isFlipped == true)
    //     flip(grayImg, grayImg,1);

    // cout << img.size() << endl;
    // imshow("Training image",img);
    // waitKey(0);

    // Check for mismatching dimensions
    // if (grayImg.cols < _hog.winSize.width)
    //     resize(grayImg,grayImg,cv::Size(_hog.winSize.width,grayImg.rows));
    // if(grayImg.rows < _hog.winSize.height)
    //     resize(grayImg,grayImg,cv::Size(grayImg.cols,_hog.winSize.height));

    //vector<float> weights;
    
    hog.compute(img, feat, Size(8,8), Size(0,0));
    //int num_features = weights.size();

    // //accumulator_set<float, stats<tag::mean, tag::moment<2> > > acc;
    // accumulator_set<float, stats<tag::max, tag::min> > acc;
    // for(int k = 0; k < num_features; ++k)
    //     acc(weights[k]);
    // // float mu = boost::accumulators::mean(acc);
    // // float std = sqrt(moment<2>(acc));
    // float xmax = boost::accumulators::max(acc);
    // float xmin = boost::accumulators::min(acc);

    // for(int i = 0; i < num_features; i++)
    // {
    //     feat.push_back(weights[i]);
    //     //feat.push_back((weights[i]-xmax)/(xmax-xmin));
    // }
    
    // Mat test;
    // renderHOG(img, test, extractedFeatures, hog.winSize, hog.cellSize, 1, 1),
    // imshow("HOG",test);
    // waitKey(0);
    //grayImg.release();
}

Mat HOGFeatureExtractor::renderHOG(Mat& img, Mat& out, vector<float>& descriptorValues, 
    Size winSize, Size cellSize, int scaleFactor, double viz_factor) const
{
    resize(img, out, Size(img.cols*scaleFactor, img.rows*scaleFactor));
 
    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14/(float)gradientBinSize; 
 
    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
 
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)            
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
 
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
 
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
 
                } // for (all bins)
 
 
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
 
            } // for (all cells)
 
 
        } // for (all block x pos)
    } // for (all block y pos)
 
 
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
 
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
 
 
    //cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
 
    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;
 
            int mx = drawX + cellSize.width/2;
            int my = drawY + cellSize.height/2;
 
            rectangle(out,
                      Point(drawX*scaleFactor,drawY*scaleFactor),
                      Point((drawX+cellSize.width)*scaleFactor,
                      (drawY+cellSize.height)*scaleFactor),
                      CV_RGB(100,100,100),
                      1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize.width/2;
                float scale = viz_factor; // just a outalization scale,
                                          // to see the lines better
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient outalization
                line(out,
                     Point(x1*scaleFactor,y1*scaleFactor),
                     Point(x2*scaleFactor,y2*scaleFactor),
                     CV_RGB(0,0,255),
                     1);
 
            } // for (all bins)
 
        } // for (cellx)
    } // for (celly)
 
 
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
      for (int x=0; x<cells_in_x_dir; x++)
      {
           delete[] gradientStrengths[y][x];            
      }
      delete[] gradientStrengths[y];
      delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
 
    return out;

}



























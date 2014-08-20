#include "Feature.h"

using namespace cv;

void FeatureExtractor::operator()(const Mat &img, Feature &feat, bool isFlipped) const
{
    this->operator()(img, feat, isFlipped);
}

void FeatureExtractor::operator()(const PascalImageDatabase &db, FeatureCollection &feats) const
{
    int n = db.getSize();

    feats.resize(n);
    for(int i = 0; i < n; i++) {
        Mat img = imread(db.getFilename(i).c_str());
        if(img.cols < 64)
            resize(img,img,Size(64,img.rows));
        if(img.rows < 128)
            resize(img,img,Size(img.cols,128));

        Rect roi = db.getRoi(i);
        bool flipped = db.isFlipped(i);
        //LOG(INFO) << " --> Extracting features from: " << db.getFilename(i) << " - " << img.size() << " - " << db.isFlipped(i);
        Mat patch = img(roi);
        (*this)(patch, feats[i], flipped);
    }
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

    if(strcasecmp(featureType.c_str(), "hog") == 0) return new HOGFeatureExtractor(params);
    // else if(strcasecmp(featureType.c_str(), "tig"    ) == 0) return new TinyImageGradFeatureExtractor(params);
    // else if(strcasecmp(featureType.c_str(), "hog"    ) == 0) return new HOGFeatureExtractor(params);

    // // Implement other features or call a feature extractor with a different set
    // // of parameters by adding more calls here.
    // else if(strcasecmp(featureType.c_str(), "custom1") == 0) throw "not implemented";
    // else if(strcasecmp(featureType.c_str(), "custom2") == 0) throw "not implemented";
    // else if(strcasecmp(featureType.c_str(), "custom3") == 0) throw "not implemented";
    else {
        throw "Unknown feature type: " + featureType;
    }
}

ParametersMap FeatureExtractor::getDefaultParameters(const std::string &featureType)
{
    ParametersMap params;
    if(strcasecmp(featureType.c_str(), "hog"     ) == 0) params = HOGFeatureExtractor::getDefaultParameters();
    // else if(strcasecmp(featureType.c_str(), "tig"    ) == 0) params = TinyImageGradFeatureExtractor::getDefaultParameters();
    // else if(strcasecmp(featureType.c_str(), "hog"    ) == 0) params = HOGFeatureExtractor::getDefaultParameters();

    // // Implement other features or call a feature extractor with a different set
    // // of parameters by adding more calls here.
    // else if(strcasecmp(featureType.c_str(), "custom1") == 0) throw "not implemented";
    // else if(strcasecmp(featureType.c_str(), "custom2") == 0) throw "not implemented";
    // else if(strcasecmp(featureType.c_str(), "custom3") == 0) throw "not implemented";
    else {
        throw "Unknown feature type: " + featureType;
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

// ============================================================================
// HOG
// ============================================================================

const char *N_ANGULAR_BINS_KEY     = "n_angular_bins";
const char *UNSIGNED_GRADIENTS_KEY = "unsigned_gradients";
const char *CELL_SIZE_KEY          = "cell_size";

ParametersMap HOGFeatureExtractor::getDefaultParameters()
{
    ParametersMap params;
    params.set(N_ANGULAR_BINS_KEY    , 18);
    params.set(UNSIGNED_GRADIENTS_KEY, 1);
    params.set(CELL_SIZE_KEY         , 6);
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

void HOGFeatureExtractor::operator()(const Mat &img, Feature &feat, bool isFlipped) const
{
    //Converting the image to grayscale
    Mat grayImg;
    cv::cvtColor(img, grayImg, CV_RGB2GRAY);

    //cout << grayImg.size() <<endl;
    resize(grayImg,grayImg,_hog.winSize);

    if(isFlipped == true)
        flip(grayImg, grayImg,1);

    vector<float> extractedFeatures;

    // Check for mismatching dimensions
    if (grayImg.cols < _hog.winSize.width)
        resize(grayImg,grayImg,cv::Size(_hog.winSize.width,grayImg.rows));
    if(grayImg.rows < _hog.winSize.height)
        resize(grayImg,grayImg,cv::Size(grayImg.cols,_hog.winSize.height));

    _hog.compute(grayImg, extractedFeatures, Size(8,8), Size(0,0));
    

    feat = Mat::zeros(extractedFeatures.size(),1, CV_32FC2);

    for(int i = 0; i < extractedFeatures.size(); i++)
    {
        feat.at<float>(i,0) = extractedFeatures[i];
    }

    extractedFeatures.clear();
    grayImg.release();
}

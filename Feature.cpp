#include "Feature.h"

Mat FeatureExtractor::renderPosNegComponents(const Feature &feat) const
{
    // Create two images, one for the positive weights and another
    // one for the negative weights
    Feature pos = Mat::zeros(feat.cols,feat.rows,CV_32FC2);
    Feature neg = Mat::zeros(feat.cols,feat.rows,CV_32FC2);
    // pos.ClearPixels();
    // neg.ClearPixels();

    // for(int y = 0; y < pos.rows; y++) {
    //     float *svmIt = (float *) feat.at<float>(0,y);
    //     float *p = (float *) pos.feat.at<float>(0, y);
    //     float *n = (float *) neg.feat.at<float>(0, y);

    //     for(int x = 0; x < pos.cols; x++, p++, n++, svmIt++) {
    //         if(*svmIt < 0) *n = fabs(*svmIt);
    //         else if(*svmIt > 0) *p = *svmIt;
    //     }
    // }

    // Mat negViz, posViz;
    // posViz = this->render(pos, true);
    // negViz = this->render(neg, true);

    // // Put positive and negative weights images side by side in a color image.
    // // Negative weights show up as red and positive weights show up as green.
    // Mat negposViz(CShape(posViz.cols * 2, posViz.rows, 3));
    // negposViz.ClearPixels();

    // for(int y = 0; y < negposViz.Shape().height; y++) {
    //     uchar *n = (uchar *) negViz.PixelAddress(0, y, 0);
    //     uchar *np = (uchar *) negposViz.PixelAddress(0, y, 2);
    //     for(int x = 0; x < negViz.Shape().width; x++, n++, np += 3) {
    //         *np = *n;
    //     }

    //     uchar *p = (uchar *) posViz.PixelAddress(0, y, 0);
    //     np = (uchar *) negposViz.PixelAddress(posViz.Shape().width, y, 1);
    //     for(int x = 0; x < negViz.Shape().width; x++, p++, np += 3) {
    //         *np = *p;
    //     }
    // }

    //return negposViz;
    return neg;
}

void FeatureExtractor::operator()(const Mat &img, Feature &feat) const
{
    this->operator()(img, feat);
}

void FeatureExtractor::operator()(const PascalImageDatabase &db, FeatureCollection &feats) const
{
    int n = db.getSize();

    feats.resize(n);
    for(int i = 0; i < n; i++) {
        //LOG(INFO) << "Extracting features from: " << db.getFilename(i);
        Mat img = imread(db.getFilename(i).c_str());
        Rect roi = db.getRoi(i);
        Mat patch = img(roi);
        (*this)(patch, feats[i]);
    }
}

// void FeatureExtractor::operator()(const SBFloatPyramid &imPyr, FeaturePyramid &featPyr) const
// {
//     featPyr.resize(imPyr.getNLevels());

//     Mat x;
//     for (int i = 0; i < imPyr.getNLevels(); i++) {
//         this->operator()(imPyr[i], featPyr[i]);
//     }
// }

Mat FeatureExtractor::render(const Feature &f, bool normalizeFeat) const
{
    // if(normalizeFeat) {
    //     Size shape = f.size();
    //     Feature fAux(shape.cols);

    //     float fMin, fMax;
    //     f.getRangeOfValues(fMin, fMax);

    //     for(int y = 0; y < shape.height; y++) {
    //         float *fIt = (float *) f.PixelAddress(0, y, 0);
    //         float *fAuxIt = (float *) fAux.PixelAddress(0, y, 0);

    //         for(int x = 0; x < shape.width * shape.nBands; x++, fAuxIt++, fIt++) {
    //             *fAuxIt = (*fIt) / fMax;
    //         }
    //     }

    //     return this->render(fAux);
    // } else {
    //     return this->render(f);
    // }
    Mat x;
    return x;
}

// std::vector<Mat> FeatureExtractor::render(const FeaturePyramid &f, bool normalizeFeat) const
// {
//     std::vector<Mat> res(f.getNLevels());
//     for(int i = 0; i < res.size(); i++) {
//         res[i] = render(f[i], normalizeFeat);
//     }
//     return res;
// }

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

    if(strcasecmp(featureType.c_str(), "ti") == 0) return new TinyImageFeatureExtractor(params);
    else if(strcasecmp(featureType.c_str(), "tig"    ) == 0) return new TinyImageGradFeatureExtractor(params);
    else if(strcasecmp(featureType.c_str(), "hog"    ) == 0) return new HOGFeatureExtractor(params);

    // Implement other features or call a feature extractor with a different set
    // of parameters by adding more calls here.
    else if(strcasecmp(featureType.c_str(), "custom1") == 0) throw CError("not implemented");
    else if(strcasecmp(featureType.c_str(), "custom2") == 0) throw CError("not implemented");
    else if(strcasecmp(featureType.c_str(), "custom3") == 0) throw CError("not implemented");
    else {
        throw CError("Unknown feature type: %s", featureType.c_str());
    }
}

ParametersMap FeatureExtractor::getDefaultParameters(const std::string &featureType)
{
    ParametersMap params;
    if(strcasecmp(featureType.c_str(), "ti"     ) == 0) params = TinyImageFeatureExtractor::getDefaultParameters();
    else if(strcasecmp(featureType.c_str(), "tig"    ) == 0) params = TinyImageGradFeatureExtractor::getDefaultParameters();
    else if(strcasecmp(featureType.c_str(), "hog"    ) == 0) params = HOGFeatureExtractor::getDefaultParameters();

    // Implement other features or call a feature extractor with a different set
    // of parameters by adding more calls here.
    else if(strcasecmp(featureType.c_str(), "custom1") == 0) throw CError("not implemented");
    else if(strcasecmp(featureType.c_str(), "custom2") == 0) throw CError("not implemented");
    else if(strcasecmp(featureType.c_str(), "custom3") == 0) throw CError("not implemented");
    else {
        throw CError("Unknown feature type: %s", featureType.c_str());
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
// TinyImage
// ============================================================================

static const char *SCALE_KEY = "scale";

ParametersMap TinyImageFeatureExtractor::getDefaultParameters()
{
    ParametersMap params;
    params.set(SCALE_KEY, 0.2);
    return params;
}

ParametersMap TinyImageFeatureExtractor::getParameters() const
{
    ParametersMap params;
    params.set(SCALE_KEY, _scale);
    return params;
}

TinyImageFeatureExtractor::TinyImageFeatureExtractor(const ParametersMap &params)
{
    _scale = params.getFloat(SCALE_KEY);
}

void TinyImageFeatureExtractor::operator()(const Mat &imgRGB, Feature &feat) const
{
    Mat tinyImg(imgRGB.cols * _scale, imgRGB.rows * _scale, CV_32FC2);

    Mat imgG;
    // convertRGB2GrayImage(imgRGB, imgG);

    // CTransform3x3 s = CTransform3x3::Scale( 1. / _scale, 1. / _scale );

    // WarpGlobal(imgG, tinyImg, s, eWarpInterpLinear);

    // feat = tinyImg;
}

Mat TinyImageFeatureExtractor::render(const Feature &f) const
{
    Mat viz;
    //TypeConvert(f, viz);
    return viz;
}

// ============================================================================
// TinyImage Gradient
// ============================================================================

ParametersMap TinyImageGradFeatureExtractor::getDefaultParameters()
{
    return TinyImageFeatureExtractor::getDefaultParameters();
}

ParametersMap TinyImageGradFeatureExtractor::getParameters() const
{
    ParametersMap params;
    params.set(SCALE_KEY, _scale);
    return params;
}

TinyImageGradFeatureExtractor::TinyImageGradFeatureExtractor(const ParametersMap &params)
{
    _scale = params.getFloat(SCALE_KEY);

    // static float derivKvals[3] = { -1, 0, 1};

    // _kernelDx.ReAllocate(CShape(3, 1, 1), derivKvals, false, 1);
    // _kernelDx.origin[0] = 1;

    // _kernelDy.ReAllocate(CShape(1, 3, 1), derivKvals, false, 1);
    // _kernelDy.origin[0] = 1;
}

void TinyImageGradFeatureExtractor::operator()(const Mat &imgRGB_, Feature &feat) const
{
    // int targetW = _round(imgRGB_.Shape().width * _scale);
    // int targetH = _round(imgRGB_.Shape().height * _scale);

    /******** BEGIN TODO ********/
    // Compute tiny image gradient feature, output should be a _targetW by _targetH
    // grayscale image, similar to tiny image. The difference here is that you will
    // compute the gradients in the x and y directions, followed by the gradient
    // magnitude.
    //
    // Steps are:
    // 1) Convert image to grayscale (see convertRGB2GrayImage in Utils.h)
    // 2) Resize image to be _targetW by _targetH
    // 3) Compute gradients in x and y directions
    // 4) Compute gradient magnitude
    //
    // Useful functions:
    // convertRGB2GrayImage, TypeConvert, WarpGlobal, Convolve

printf("TODO: %s:%d\n", __FILE__, __LINE__); 

    /******** END TODO ********/
}

Mat TinyImageGradFeatureExtractor::render(const Feature &f) const
{
    Mat viz;
    //TypeConvert(f, viz);
    return viz;
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

    //static float derivKvals[3] = { -1, 0, 1};

    // _kernelDx.ReAllocate(CShape(3, 1, 1), derivKvals, false, 1);
    // _kernelDx.origin[0] = 1;

    // _kernelDy.ReAllocate(CShape(1, 3, 1), derivKvals, false, 1);
    // _kernelDy.origin[0] = 1;

    // Visualization Stuff
    // A set of patches representing the bin orientations. When drawing a hog cell
    // we multiply each patch by the hog bin value and add all contributions up to
    // form the visual representation of one cell. Full HOG is achieved by stacking
    // the viz for individual cells horizontally and vertically.
    _oriMarkers.resize(_nAngularBins);
    const int ms = 11;
    CShape markerShape(ms, ms, 1);

    // FIXME: add patches for contrast sensitive dimensions (half filled circle)

    // First patch is a horizontal line
    _oriMarkers[0].ReAllocate(markerShape, true);
    _oriMarkers[0].ClearPixels();
    for(int i = 1; i < ms - 1; i++) _oriMarkers[0].Pixel(/*floor(*/ ms / 2 /*)*/, i, 0) = 1;

    // The other patches are obtained by rotating the first one
    CTransform3x3 T = CTransform3x3::Translation((ms - 1) / 2.0, (ms - 1) / 2.0);
    for(int angBin = 1; angBin < _nAngularBins; angBin++) {
        double theta;
        if(_unsignedGradients) theta = 180.0 * (double(angBin) / _nAngularBins);
        else theta = 360.0 * (double(angBin) / _nAngularBins);
        CTransform3x3 R  = T * CTransform3x3::Rotation(theta) * T.Inverse();

        _oriMarkers[angBin].ReAllocate(markerShape, true);
        _oriMarkers[angBin].ClearPixels();

        WarpGlobal(_oriMarkers[0], _oriMarkers[angBin], R, eWarpInterpLinear);
    }
}

void HOGFeatureExtractor::operator()(const Mat &img, Feature &feat) const
{
    /******** BEGIN TODO ********/
    // Compute the Histogram of Oriented Gradients feature
    //
    // Steps are:
    // 1) Compute gradients in x and y directions. We provide the
    //    derivative kernel proposed in the paper in _kernelDx and
    //    _kernelDy.
    // 2) Compute gradient magnitude and orientation
    // 3) Add contribution each pixel to HOG cells whose
    //    support overlaps with pixel. The contribution should
    //    be weighted by a gaussian centered at the corresponding
    //    HOG cell. Each cell has a support of size
    //    _cellSize and each histogram has _nAngularBins. Note that
    //    pixels away from the borders of the image should contribute to
    //    at least four HOG cells.
    // 4) Normalize HOG for each cell. One simple strategy that is
    //    is also used in the SIFT descriptor is to first threshold
    //    the bin values so that no bin value is larger than some
    //    threshold (we leave it up to you do find this value) and
    //    then re-normalize the histogram so that it has norm 1. A more
    //    elaborate normalization scheme is proposed in Dalal & Triggs
    //    paper but we leave that as extra credit.
    //
    // Useful functions:
    // convertRGB2GrayImage, TypeConvert, WarpGlobal, Convolve

    //Converting the image to 
    Mat grayImg;
    cv::cvtColor(img, grayImg, CV_RGB2GRAY);
    resize(grayImg,grayImg,_hog.winSize);

    vector<float> extractedFeatures;

    // Check for mismatching dimensions
    if (grayImg.cols != _hog.winSize.width || grayImg.rows != _hog.winSize.height) {
       extractedFeatures.clear();
       throw CError("Error in image dimensions");
    }
    
    _hog.compute(grayImg, extractedFeatures, Size(8,8), Size(0,0));

    feat = Mat::zeros(extractedFeatures.size(),1, CV_32F);

    for(int i = 0; i < extractedFeatures.size(); i++)
    {
        feat.at<float>(i,0) = extractedFeatures[i];
    }

    grayImg.release();

    /******** END TODO ********/
}

Mat HOGFeatureExtractor::render(const Feature &f) const
{
    // CShape cellShape = _oriMarkers[0].Shape();
    // CFloatImage hogImgF(CShape(cellShape.width * f.Shape().width, cellShape.height * f.Shape().height, 1));
    // hogImgF.ClearPixels();

    // float minBinValue, maxBinValue;
    // f.getRangeOfValues(minBinValue, maxBinValue);

    // // For every cell in the HOG
    // for(int hi = 0; hi < f.Shape().height; hi++) {
    //     for(int hj = 0; hj < f.Shape().width; hj++) {

    //         // Now _oriMarkers, multiplying contribution by bin level
    //         for(int hc = 0; hc < _nAngularBins; hc++) {
    //             float v = f.Pixel(hj, hi, hc) / maxBinValue;
    //             for(int ci = 0; ci < cellShape.height; ci++) {
    //                 float *cellIt = (float *) _oriMarkers[hc].PixelAddress(0, ci, 0);
    //                 float *hogIt = (float *) hogImgF.PixelAddress(hj * cellShape.height, hi * cellShape.height + ci, 0);

    //                 for(int cj = 0; cj < cellShape.width; cj++, hogIt++, cellIt++) {
    //                     (*hogIt) += v * (*cellIt);
    //                 }
    //             }
    //         }

    //     }
    // }

    Mat hogImg;
    //TypeConvert(hogImgF, hogImg);
    return hogImg;
}



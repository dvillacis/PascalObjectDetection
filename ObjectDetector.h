#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include "Common.h"
#include "Detection.h"
#include "SubBandImagePyramid.h"

class ObjectDetector
{
public:
    static ParametersMap getDefaultParameters();
    ParametersMap getParameters() const;

    ObjectDetector(const ParametersMap &params = getDefaultParameters());
    ObjectDetector(int winSizeNMS,  double respThresh, double overlapThresh);

    void operator()( const CFloatImage &svmResp, const Size &roiSize, double featureScaleFactor, std::vector<Detection> &dets, double imScale = 1.0 ) const;
    void operator()( const SBFloatPyramid &svmRespPyr, const Size &roiSize, double featureScaleFactor, std::vector<Detection> &dets ) const;

private:
    int _winSizeNMS;
    double _respThresh;
    double _overlapThresh;
};

#endif // OBJECT_DETECTOR_H
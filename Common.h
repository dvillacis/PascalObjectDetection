#ifndef COMMON_H
#define COMMON_H

// Standard library includes
#define _USE_MATH_DEFINES // To get M_PI back on windows

#include <algorithm>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <set>

#include <cmath>
#include <cassert>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <stdint.h>
// Library for Support Vector Machine algo
#include "libsvm/svm.h"

// Boost 
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

// Google Log
#include <glog/logging.h>

// Open CV
#include <opencv2/opencv.hpp>

// ============================================================================
// Persistence
// ============================================================================
#define OBJECT_DETECTOR_KEY   "ObjectDetector"
#define IMAGE_PYRAMID_KEY     "ImagePyramid"
#define FEATURE_EXTRACTOR_KEY "FeatureExtractor"
#define FEATURE_TYPE_KEY      "feature_type"
#define SVM_CONFIG_KEY	      "svm_config"

#endif // COMMON_H


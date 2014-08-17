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
#include "libsvm-3.14/svm.h"

// Boost 
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

// Google Log
#include <glog/logging.h>

// Open CV
#include <opencv2/opencv.hpp>

// Image Lib
//#include "ImageLib/ImageLib.h"



// ============================================================================
// Needed by Windows
// ============================================================================
#if defined(_WIN32) || defined(_WIN64)
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#define strcasecmp _stricmp
#define strncasecmp _strnicmp

#define NOMINMAX // to get min and max back in windows
#ifndef GL_BGR
#define GL_BGR 0x80E0
#endif

#ifndef GL_BGRA
#define GL_BGRA 0x80E1
#endif

#endif

// ============================================================================
// GUI
// ============================================================================

// Spacing (in pixels) between elements in the GUI
static const int DEFAULT_BORDER = 5;
static const int BOX_WIDTH = 2; // Border around the box that Fl draws (guessing)

// ============================================================================
// Persistence
// ============================================================================
#define OBJECT_DETECTOR_KEY   "ObjectDetector"
#define IMAGE_PYRAMID_KEY     "ImagePyramid"
#define FEATURE_EXTRACTOR_KEY "FeatureExtractor"
#define FEATURE_TYPE_KEY      "feature_type"
#define SVM_CONFIG_KEY	      "svm_config"

#endif // COMMON_H


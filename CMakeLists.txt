PROJECT(objdet) 

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Where to search for cmake scripts
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Check for dependencies (Ref: http://www.cmake.org/Wiki/CMakeForFLTK)
# http://stackoverflow.com/questions/9779901/setting-up-fltk-on-windows-with-cmake

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(JPEG REQUIRED)
FIND_PACKAGE(GLOG REQUIRED)
SET(Boost_USE_STATIC_LIBS OFF) 
SET(Boost_USE_MULTITHREADED ON)  
SET(Boost_USE_STATIC_RUNTIME OFF) 
FIND_PACKAGE(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
IF(Boost_FOUND)
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	MESSAGE(status " Boost libs: " ${Boost_LIBRARIES})
ENDIF()

LIST(APPEND CMAKE_CXX_FLAGS "-Wall")

FIND_PACKAGE(OpenGL REQUIRED)

# # Build subdirectories
ADD_SUBDIRECTORY(thirdparty/ImageLib)
INCLUDE_DIRECTORIES(thirdparty/)

ADD_SUBDIRECTORY(thirdparty/libsvm-3.14)

ADD_LIBRARY(od
	Feature.h                                           Feature.cpp 
	SupportVectorMachine.h                              SupportVectorMachine.cpp 
	PascalImageDatabase.h                               PascalImageDatabase.cpp 
	ImageDatabase.h                                     ImageDatabase.cpp 
	PrecisionRecall.h                                   PrecisionRecall.cpp
	ObjectDetector.h                                    ObjectDetector.cpp
	Detection.h                                         Detection.cpp   
	FileIO.h                                            FileIO.cpp
	ParametersMap.h                                     ParametersMap.cpp
	Common.h    
)

TARGET_LINK_LIBRARIES(od image svm jpegrw ${JPEG_LIBRARY} ${OpenCV_LIBS})

# # Building the project 
ADD_EXECUTABLE(objdet main.cpp)
TARGET_LINK_LIBRARIES(objdet od ${JPEG_LIBRARY} ${OPENGL_LIBRARIES} ${Boost_LIBRARIES} ${GLOG_LIB_1} ${GLOG_LIB_2} ${GLOG_LIB_3} ${GLOG_LIB_4} ${GLOG_LIB_5} ${GLOG_LIB_6} ${GLOG_LIB_7})

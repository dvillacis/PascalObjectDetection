#ifndef PASCAL_IMAGE_DATABASE_H
#define PASCAL_IMAGE_DATABASE_H

#include "Common.h"

using namespace std;

//! Pascal Image Database Class
/*!
    This class stores the list of images provided in the PASCAL VOC format, 
    this data will later be used to train and test a SVM. This class contains a 
    list of image filenames, extracted rectangles defining the ground truth along with their predicted value.

    Here we use all the procedures defined in the Dalal and Triggs paper to improve generalization, 
    as flipping the true positives and generating random cuts from the negative samples.
*/

class PascalImageDatabase
{
private:
    vector<string> _filenames;
    vector<float> _labels;
    vector<bool> _flipped;
    int _positivesCount;
    int _negativesCount;
    string _dbFilename;
    string _category;

    // What is the size of the object in the image (object is assumed to be
    // centered and of the same size in all images)
    vector<cv::Rect> _rois;

    bool getROI(string imageName, vector<cv::Rect>& rois, vector<float>& roiLabels);


public:
    //! Constructor
    PascalImageDatabase();
    
    //! Constructor
    /*!
        This constructor takes the filename of the database to use and the name of the category that will be trained
    */
    PascalImageDatabase(const char *dbFilename, const string category);

    //! Constructor
    /*!
        This constructor generates the database with the givel labels and filenames provided directly.
    */
    PascalImageDatabase(const vector<float> &labels, const vector<string> &filenames);

    //! Load a database from a file
    void load(const char *dbFilename);

    //! Save the database to a file
    void save(const char *dbFilename);

    // Accessors
    //! Accessor to get the label of a specific sample
    const int getLabel(int idx) const { return _labels[idx]; }

    //! Accessor to get all the labels defined in the database
    const vector<float> &getLabels() const { return _labels; }

    //! Accessor to get the filename of a specific sample
    const string getFilename(int idx) const { return _filenames[idx]; }

    //! Accessor to get all the filenames defined in the database
    const vector<string> &getFilenames() const { return _filenames; }

    //! Accesor to get the Region of Interest of a specific sample
    const cv::Rect getRoi(int idx) const { return _rois[idx]; }

    //! Accesor to get all the Region of Interest defined in the database
    const vector<cv::Rect> getRois() const { return _rois; }

    //! Accesor to the property that specifies if the sample needs to be flipped
    const bool isFlipped(int idx) const { return _flipped[idx]; }

    //! Accesor get a list of all the samples flipped configuration
    const vector<bool> getFlipped() const { return _flipped; }

    // Info about the database
    //! Get the number of positive samples found in the database
    int getPositivesCount() const { return _positivesCount; }

    //! Get the number of negative samples found in the database
    int getNegativesCount() const { return _negativesCount; }

    //! Get the number of unlabeled samples found in the database
    int getUnlabeledCount() const { return _labels.size() - _positivesCount - _negativesCount; }

    //! Get the size of the database
    int getSize() const { return _labels.size(); }

    //! Get the file used to create the database
    string getDatabaseFilename() const { return _dbFilename; }
    
};

//! Prints information about the dataset
ostream &operator<<(ostream &s, const PascalImageDatabase &db);

#endif // PASCAL_IMAGE_DATABASE_H
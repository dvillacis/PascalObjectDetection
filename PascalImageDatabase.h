#ifndef PASCAL_IMAGE_DATABASE_H
#define PASCAL_IMAGE_DATABASE_H

#include "Common.h"

using namespace std;

// Stores the datasets used for training and testing the Support
// Vector Machine class. Contains basically a list of image filenames
// together with their true or predicted labels.
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

    void getROI(string imageName, vector<cv::Rect>& rois, vector<float>& roiLabels);


public:
    // Create a new database.
    PascalImageDatabase();
    PascalImageDatabase(const char *dbFilename, const string category);
    PascalImageDatabase(const vector<float> &labels, const vector<string> &filenames);

    // Load a database from file.
    void load(const char *dbFilename);
    void save(const char *dbFilename);

    // Accessors
    const int getLabel(int idx) const { return _labels[idx]; }
    const vector<float> &getLabels() const { return _labels; }
    const string getFilename(int idx) const { return _filenames[idx]; }
    const vector<string> &getFilenames() const { return _filenames; }
    const cv::Rect getRoi(int idx) const { return _rois[idx]; }
    const vector<cv::Rect> getRois() const { return _rois; }
    const bool isFlipped(int idx) const { return _flipped[idx]; }
    const vector<bool> getFlipped() const { return _flipped; }

    // Info about the database
    int getPositivesCount() const { return _positivesCount; }
    int getNegativesCount() const { return _negativesCount; }
    int getUnlabeledCount() const { return _labels.size() - _positivesCount - _negativesCount; }
    int getSize() const { return _labels.size(); }
    string getDatabaseFilename() const { return _dbFilename; }
    
};

// Prints information about the dataset
ostream &operator<<(ostream &s, const PascalImageDatabase &db);

#endif // PASCAL_IMAGE_DATABASE_H
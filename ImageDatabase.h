#ifndef IMAGE_DATABASE_H
#define IMAGE_DATABASE_H

#include "Common.h"
#include "Detection.h"

using namespace std;

//! Pascal Image Database Class
/*!
    This class stores the list of images provided in the PASCAL VOC format.
*/

class ImageDatabase
{
private:
    vector<string> _filenames;
    vector<vector<Detection> > _detections;
    string _dbFilename;
    string _category;

    int _positivesCount;
    int _negativesCount;

    vector<float> _labels;

public:
    //! Constructor
    ImageDatabase();

    //! Constructor
    /*!
        \param dbFilename Path where the input image list is located.
        \param category  Class that we want to train.
    */
    ImageDatabase(const string &dbFilename, const string category);
    ImageDatabase(const vector<vector<Detection> > &dets, const vector<string> &fnames);

    //! Load a database from file.
    /*!
        \dbFilename Path where the input image list is located.
    */
    void load(const string &dbFilename);

    //! Save a database to file.
    /*!
        \dbFilename Path where the input image list is located.
    */
    void save(const string &dbFilename);

    // Accessors
    const vector<vector<Detection> > &getDetections() const { return _detections; }
    const vector<string> &getFilenames() const { return _filenames; }

    // Info about the database
    int getPositivesCount() const { return _positivesCount; }
    int getNegativesCount() const { return _negativesCount; }
    int getUnlabeledCount() const { return _labels.size() - _positivesCount - _negativesCount; }
    int getSize() const { return _detections.size(); }
    string getDatabaseFilename() const { return _dbFilename; }

    // Getting the ground truth from the annotations
    vector<Detection> getGroundTruth(string imageName);
};

// Prints information about the dataset
ostream &operator<<(ostream &s, const ImageDatabase &db);

#endif // IMAGE_DATABASE_H
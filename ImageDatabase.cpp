#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include "PascalAnnotation.h"
#include "ImageDatabase.h"


using namespace std;

static const char *SIGNATURE = "ImageDataset";

// Image Database class

ImageDatabase::ImageDatabase()
{
}

ImageDatabase::ImageDatabase(const string &dbFilename)
{
    _category = "car";
    load(dbFilename);
}

ImageDatabase::ImageDatabase(const vector<vector<Detection> > &dets,
                             const vector<string> &fnames):
    _detections(dets),
    _filenames(fnames)
{
}

vector<Detection> ImageDatabase::getGroundTruth(string imageName)
{
    vector<Detection> dets;

    vector<string> parts;
    boost::split(parts,imageName,boost::is_any_of("/."),boost::token_compress_on);
    string annotationsPath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/Annotations/";
    string annotationsFilename = annotationsPath + parts[parts.size()-2] + ".xml";

    pascal_annotation annotation;
    annotation.load(annotationsFilename);

    for(int i = 0; i < annotation.objects.size(); ++i){
        if(boost::iequals(annotation.objects[i].name,_category))
        {
            Detection det;
            LOG(INFO) << "Obtained gt for " << imageName;
        }
    }

    return dets;
}

void ImageDatabase::load(const string &dbFilename)
{
    string imagePath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/JPEGImages/";

    _dbFilename = dbFilename;

    ifstream f(dbFilename.c_str());
    if(!f.is_open()) {
        throw "Could not open file " + dbFilename + " for reading";
    }
    else
    {
        string line;
        while(getline(f, line)){
            vector<string> parts;
            boost::split(parts, line, boost::is_any_of(" "), boost::token_compress_on);
            string imageName = imagePath + parts[0] + ".jpg";
            _filenames.push_back(imageName);
            vector<Detection> detections = getGroundTruth(imageName);
            _detections.push_back(detections);
        }
    }

    // char sig[200];
    // f.read(sig, strlen(SIGNATURE));
    // sig[strlen(SIGNATURE)] = '\0';
    // if (strcmp(sig, SIGNATURE) != 0) {
    //     throw "Bad signature for file, expecting \"%s\" but got \"%s\"", SIGNATURE, sig);
    // }

    // int nItems;
    // f >> nItems;

    // assert(nItems > 0);

    // _filenames.resize(nItems);
    // _detections.resize(nItems);
    // for(int i = 0; i < nItems; i++) {
    //     f >> _filenames[i];

    //     int nDets = 0;
    //     f >> nDets;

    //     _detections[i].resize(nDets);
    //     for (int j = 0; j < nDets; j++) f >> _detections[i][j];
    // }
}

void ImageDatabase::save(const string &dbFilename)
{
    ofstream f(dbFilename.c_str());
    if(!f.is_open()) {
        throw "Could not open file " + dbFilename + " for writing";
    }

    f << SIGNATURE << "\n";
    f << getSize() << "\n";

    for(int i = 0; i < getSize(); i++) {
        f << _filenames[i] << " ";
        f << _detections[i].size() << " ";
        for (int j = 0; j < _detections[i].size(); j++) {
            f << _detections[i][j] << " ";
        }
        f << "\n";
    }
}

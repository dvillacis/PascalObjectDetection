#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include "PascalAnnotation.h"
#include "ImageDatabase.h"


using namespace std;

static const char *SIGNATURE = "ImageDataset";

// Image Database class

ImageDatabase::ImageDatabase():
    _positivesCount(0), 
    _negativesCount(0)
{
}

ImageDatabase::ImageDatabase(const string &dbFilename, const string category):
    _positivesCount(0), 
    _negativesCount(0)
{
    _category = category;
    load(dbFilename);
}

ImageDatabase::ImageDatabase(const vector<vector<Detection> > &dets,
                             const vector<string> &fnames):
    _detections(dets),
    _filenames(fnames),
    _positivesCount(0), 
    _negativesCount(0)
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
            cv::Rect r(annotation.objects[i].bndbox.xmin,
                annotation.objects[i].bndbox.ymin,
                annotation.objects[i].bndbox.xmax - annotation.objects[i].bndbox.xmin,
                annotation.objects[i].bndbox.ymax - annotation.objects[i].bndbox.ymin);

            Detection det(r,1);
            dets.push_back(det);
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
        throw std::runtime_error("Could not open file " + dbFilename + " for reading");
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

            int label = -1;
            if(atof(parts[1].c_str()) > 0)
                label = 1;
            _labels.push_back(label);

            if(label < 0) _negativesCount++;
                else if(label > 0) _positivesCount++;
        }
    }
}

void ImageDatabase::save(const string &dbFilename)
{
    ofstream f(dbFilename.c_str());
    if(!f.is_open()) {
        throw std::runtime_error("Could not open file " + dbFilename + " for writing");
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

ostream & operator << (ostream &s, const ImageDatabase &db)
{
    s << "DATABASE INFO\n"
      << setw(20) << "Original filename:" << " " << db.getDatabaseFilename() << "\n"
      << setw(20) << "Positives:" << setw(5) << right << db.getPositivesCount() << "\n"
      << setw(20) << "Negatives:"   << setw(5) << right << db.getNegativesCount() << "\n"
      << setw(20) << "Unlabeled:"  << setw(5) << right << db.getUnlabeledCount() << "\n"
      << setw(20) << "Total:"      << setw(5) << right << db.getSize() << "\n";

    return s;
}

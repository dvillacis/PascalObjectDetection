#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

#include "PascalAnnotation.h"
#include "PascalImageDatabase.h"

using namespace std;


//Pascal Image Database class

PascalImageDatabase::PascalImageDatabase():
    _positivesCount(0), _negativesCount(0)
{
}

PascalImageDatabase::PascalImageDatabase(const char *dbFilename, string category):
    _positivesCount(0), _negativesCount(0)
{
    _category = category;
    load(dbFilename);
}

PascalImageDatabase::PascalImageDatabase(const vector<float> &labels, const vector<string> &filenames):
    _positivesCount(0), _negativesCount(0)
{
    _labels = labels;
    _filenames = filenames;

    for(vector<float>::iterator i = _labels.begin(); i != _labels.end(); i++) {
        if(*i > 0) _positivesCount++;
        else if(*i < 0) _negativesCount++;
    }
}

void PascalImageDatabase::getROI(string imageName, vector<cv::Rect>& rois, vector<float>& roiLabels)
{
    vector<string> parts;
    boost::split(parts,imageName,boost::is_any_of("/."),boost::token_compress_on);

    string annotationsPath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/Annotations/";
    string annotationsFilename = annotationsPath + parts[parts.size()-2] + ".xml";

    pascal_annotation annotation;
    annotation.load(annotationsFilename);

    for(int i = 0; i < annotation.objects.size(); ++i){
        cv::Rect r(annotation.objects[i].bndbox.xmin,
            annotation.objects[i].bndbox.ymin,
            annotation.objects[i].bndbox.xmax - annotation.objects[i].bndbox.xmin,
            annotation.objects[i].bndbox.ymax - annotation.objects[i].bndbox.ymin);
        rois.push_back(r);

        int label = -1;
        if(boost::iequals(annotation.objects[i].name,_category))
            label = 1;
        roiLabels.push_back(label);
    }
}

void PascalImageDatabase::load(const char *dbFilename)
{
    string imagePath = "/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/JPEGImages/";

    _dbFilename = string(dbFilename);

    _negativesCount = 0;
    _positivesCount = 0;

    ifstream f(dbFilename);
    if(!f.is_open()) {
        throw "Could not open file " + (string)dbFilename + " for reading";
    }
    else{
        string line;
        int i = 0;
        while(getline(f,line)){
            if(i > 100)
                break;
            vector<string> parts;
            boost::split(parts,line,boost::is_any_of(" "),boost::token_compress_on);

            float label = atof(parts[1].c_str());
            string imageName = imagePath + parts[0] + ".jpg";

            vector<cv::Rect> roi;
            vector<float> roiLabels;
            getROI(imageName, roi, roiLabels);

            for(int i = 0; i < roi.size(); ++i) {
                _filenames.push_back(imageName);
                _rois.push_back(roi[i]);
                _labels.push_back(roiLabels[i]);

                if(roiLabels[i] < 0) _negativesCount++;
                else if(roiLabels[i] > 0) _positivesCount++;

                //LOG(INFO) << imageName << " - " << roi[i] << " (" << roiLabels[i] << ")";
            }
            //i++;
        }
    }
}

void
PascalImageDatabase::save(const char *dbFilename)
{
    ofstream f(dbFilename);
    if(!f.is_open()) {
        throw "Could not open file " + (string)dbFilename + " for writing";
    }

    for(int i = 0; i < _labels.size(); i++) {
        f << _labels[i] << " " << _filenames[i] << "\n";
    }
}

ostream & operator << (ostream &s, const PascalImageDatabase &db)
{
    s << "DATABASE INFO\n"
      << setw(20) << "Original filename:" << " " << db.getDatabaseFilename() << "\n"
      << setw(20) << "Positives:" << setw(5) << right << db.getPositivesCount() << "\n"
      << setw(20) << "Negatives:"   << setw(5) << right << db.getNegativesCount() << "\n"
      << setw(20) << "Unlabeled:"  << setw(5) << right << db.getUnlabeledCount() << "\n"
      << setw(20) << "Total:"      << setw(5) << right << db.getSize() << "\n";

    return s;
}
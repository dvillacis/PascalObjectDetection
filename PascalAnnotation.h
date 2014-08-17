#ifndef PASCAL_ANNOTATION_H
#define PASCAL_ANNOTATION_H

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

using namespace std;

// Defining PASCAL Annotation 
struct source_annotation
{
    string database;
    string annotation;
    string image;
    string flickrid;
};

struct owner_annotation
{
    string flickrid;
    string name;
};

struct size_annotation
{
    int width;
    int height;
    int depth;
};

struct bndbox_annotation
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};

struct object_annotation
{
    string name;
    string pose;
    bool truncated;
    bool difficult;
    bndbox_annotation bndbox;
};

struct pascal_annotation
{
    string folder;
    string filename;
    bool segmented;
    source_annotation source;
    owner_annotation owner;
    size_annotation size;
    vector<object_annotation> objects;

    void load(const string &annotationsFilename);
};

inline void pascal_annotation::load(const string &annotationsFilename)
{
    using boost::property_tree::ptree;
    ptree pt;
    read_xml(annotationsFilename,pt);
    folder = pt.get<string>("annotation.folder");
    filename = pt.get<string>("annotation.filename");
    segmented = pt.get<bool>("annotation.segmented");
    source.database = pt.get<string>("annotation.source.database");
    source.annotation = pt.get<string>("annotation.source.annotation");
    source.image = pt.get<string>("annotation.source.image");
    source.flickrid = pt.get<string>("annotation.source.flickrid");
    owner.flickrid = pt.get<string>("annotation.owner.flickrid");
    owner.name = pt.get<string>("annotation.owner.name");
    size.width = pt.get<int>("annotation.size.width");
    size.height = pt.get<int>("annotation.size.height");
    size.depth = pt.get<int>("annotation.size.depth");

    BOOST_FOREACH(ptree::value_type const& v, pt.get_child("annotation")) {
        if(v.first == "object"){
            object_annotation object;
            object.name = v.second.get<string>("name");
            object.pose = v.second.get<string>("pose");
            object.truncated = v.second.get<bool>("truncated");
            object.difficult = v.second.get<bool>("difficult");
            object.bndbox.xmin = v.second.get<int>("bndbox.xmin");
            object.bndbox.ymin = v.second.get<int>("bndbox.ymin");
            object.bndbox.xmax = v.second.get<int>("bndbox.xmax");
            object.bndbox.ymax = v.second.get<int>("bndbox.ymax");
            objects.push_back(object);
        }
    }
}

#endif // PASCAL_ANNOTATION_H
#ifndef PASCAL_ANNOTATION_H
#define PASCAL_ANNOTATION_H

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

using namespace std;

// Defining PASCAL Annotation 

//! Pascal SOURCE node annotation
struct source_annotation
{
    string database;
    string annotation;
    string image;
    string flickrid;
};

//! Pascal OWNER node annotation
struct owner_annotation
{
    //! Flick ID of the image's author
    string flickrid; 
    //! Name of the image's author
    string name; 
};

//! Pascal SIZE node annotation
struct size_annotation
{
    //! Widht of the original image
    int width; 
    //! Height of the original image
    int height; 
    //! Depth of the original image
    int depth; 
};

//! Pascal BNDBOX node annotation
struct bndbox_annotation
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};

//! Pascal OBJECT node annotation
struct object_annotation
{
    //! Category name of the object
    string name;
    //! Pose of the object in the picture
    string pose;
    //! Define if the object is truncated in the picture
    bool truncated;
    bool difficult;
    //! Bounding Box containing the object in the image 
    bndbox_annotation bndbox;
};

//! Pascal PASCAL node annotation
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

//! Load the PASCAL XML Annotation
/*!
    \param annotationsFilename Path where the annotation file for the image is located.
*/
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
#ifndef PARAMETERS_MAP
#define PARAMETERS_MAP

#include "Common.h"

//! Parameters Map
/*!
    This is a generic class that allos to load and save configuration files, 
    this files contain the parameters used to initialize the SVM Parameters and the Object Detection Parameters.
*/

class ParametersMap : public std::map<std::string, std::string>
{
public:
    //! Constructor
    ParametersMap() {};

    //! Set a key entry with a double value
    void set(const std::string &key, double val);

    //! Set a key entry with an integer value
    void set(const std::string &key, int val);

    //! Set a key entry with a string value
    void set(const std::string &key, const std::string &val);

    //! Get the value referenced to a key and cast it to an integer
    int getInt(const std::string &key) const;

    //! Get the value referenced to a key and cast it to a float
    double getFloat(const std::string &key) const;

    //! Get the value referenced to a key and cast it to a string
    const std::string &getStr(const std::string &key) const;

    //! Save the parameter file named fname to the hard drive
    void save(const std::string &fname) const;

    //! Low level function to write the file to disk
    void save(FILE *f) const;

    //! Low level function to load a file
    void load(FILE *f);
};

//! Save dictionary of paramater maps to file
/*!
    \param fname Path of the parameter file to be saved
    \param params Map containing the parameters information
*/
void saveToFile(const std::string &fname, const std::map<std::string, ParametersMap> &params);

//! Load dictionary of paramater maps from file
/*!
    \param fname Path of the parameter file to be saved
    \param params Map containing the parameters information
*/
void loadFromFile(const std::string &fname, std::map<std::string, ParametersMap> &params);

#endif // PARAMETERS_MAP
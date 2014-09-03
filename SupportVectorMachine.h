#ifndef SUPPORT_VECTOR_MACHINE_H
#define SUPPORT_VECTOR_MACHINE_H

#include "Common.h"
#include "Feature.h"
#include "PascalImageDatabase.h"

//! Support Vector Machine Class
/*!
    This class is a wrapper for LIBSVM that allows the SVM training from the image database.
    It creates a primal form of the weights so it can be used along with the OPENCV framework.
*/

class SupportVectorMachine
{
private:
    struct svm_parameter _param;     // set by parse_command_line
    struct svm_problem _prob;        // set by read_problem
    struct svm_model *_model;
    struct svm_node *_x_space;

    svm_node *_data;

private:
    //! De allocate memory
    void _deinit();

public:
    //! Constructor
    SupportVectorMachine();

    //! Loads SVM with user-defined parameters
    /*!
        \param params ParameterMap containing the SVM configuration
    */
    SupportVectorMachine(const ParametersMap &params);

    //! Loads SVM model from file
    /*!
        \param modelFName Path to a file containing the SVM configuration
    */
    SupportVectorMachine(const std::string &modelFName);

    //! Destructor
    ~SupportVectorMachine();

    //! Train the SVM model
    void train(const std::vector<float> &labels, FeatureCollection &features, std::string svmModelFName);

    //! Predict the decision value of a feature
    /*! 
        Run classifier on feature, size of feature must match one used for model training.
        \param feature HOG calculated features from an image
    */ 
    float predict(const Feature &feature) const;

    //! Predict the decision value of a feature
    /*! 
        Run classifier on feature, size of feature must match one used for model training.
        \param feature HOG calculated features from an image
    */ 
    //float predict(const vector<float> &feature) const;

    //! Predict the label of a feature
    /*! 
        Run classifier on feature, size of feature must match one used for model training.
        \param feature HOG calculated features from an image
    */
    float predictLabel(const Feature &feature) const;
    //! Predict the label of a feature
    /*! 
        Run classifier on feature, size of feature must match one used for model training.
        \param feature HOG calculated features from an image
    */
    float predictLabel(const vector<float> &feature, double& decisionValue) const;

    //! Gets a collection of predictions given a collection of features
    std::vector<float> predict(const FeatureCollection &fset);
    std::vector<float> predictLabel(const FeatureCollection &fset) const;

    //! Get the primal form for the svm
    std::vector<float> getDetector() const;

    //! Print the parameters chosen for the SVM
    void printSVMParameters();

    //! Get SVM weights in the shape of the original features
    //vector<float> getWeights() const;
    double getBiasTerm() const;

    //! Get default parameters
    static ParametersMap getDefaultParameters();
    ParametersMap getParameters();

    //Mat renderSVMWeights(const FeatureExtractor *featExtractor);

    //! Load model to file
    /*!
        \param filename Path where the configuration file is located.
    */
    svm_model * load(const std::string &filename);

    //! Save model to file
    /*!
        \param filename Path where the configuration file will be located.
    */
    void save(const std::string &filename) const;

    //! Verify if the svm is initiallized
    bool initialized() const { return _model != NULL; }
};

#endif // SUPPORT_VECTOR_MACHINE_H
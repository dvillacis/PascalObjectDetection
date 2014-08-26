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
    //! LIBSVM model
    struct svm_model *_model;

    //! LIBSVM node definition
    svm_node *_data; // Have to keep this around if we want to save the model after training
    cv::Size _fVecShape; // Shape of feature vector

    //! SVM Parameters
    svm_parameter _parameter;

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
    void train(const std::vector<float> &labels, const FeatureCollection &fset);

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
    std::vector<float> predict(const FeatureCollection &fset) const;
    std::vector<float> predictLabel(const FeatureCollection &fset) const;

    //! Get the primal form for the svm
    std::vector<float> getDetector() const;

    // Runs classifier at every location of feature feat, returns a
    // single channel image with classifier output at each location.
    void predictSlidingWindow(const Feature &feat, Mat &response) const;

    // Runs classifier on each level of the pyramid, returns a pyramid
    // where each level contains the response of the classifier at the
    // corresponding level of the input pyramid.
    void predictSlidingWindow(const FeatureCollection &featPyr, vector<Mat> &responsePyr) const;

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
    void load(const std::string &filename);
    void load(FILE *fp);

    //! Save model to file
    /*!
        \param filename Path where the configuration file will be located.
    */
    void save(const std::string &filename) const;
    void save(FILE *fp) const;

    //! Verify if the svm is initiallized
    bool initialized() const { return _model != NULL; }
};

#endif // SUPPORT_VECTOR_MACHINE_H
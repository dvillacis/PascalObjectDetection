#ifndef SUPPORT_VECTOR_MACHINE_H
#define SUPPORT_VECTOR_MACHINE_H

#include "Common.h"
#include "Feature.h"
#include "PascalImageDatabase.h"

class SupportVectorMachine
{
private:
    struct svm_model *_model;
    svm_node *_data; // Have to keep this around if we want to save the model after training
    cv::Size _fVecShape; // Shape of feature vector

    // SVM Parameters
    svm_parameter _parameter;

private:
    // De allocate memory
    void _deinit();

public:
    SupportVectorMachine();

    // Loads SVM with user-defined parameters
    SupportVectorMachine(const ParametersMap &params);

    // Loads SVM model from file
    SupportVectorMachine(const std::string &modelFName);
    ~SupportVectorMachine();

    void train(const std::vector<float> &labels, const FeatureCollection &fset);

    // Run classifier on feature, size of feature must match one used for
    // model training
    float predict(const Feature &feature) const;
    float predictLabel(const Feature &feature) const;
    std::vector<float> predict(const FeatureCollection &fset) const;

    // Get the primal for the svm
    std::vector<float> getDetector() const;

    // Runs classifier at every location of feature feat, returns a
    // single channel image with classifier output at each location.
    void predictSlidingWindow(const Feature &feat, Mat &response) const;

    // Runs classifier on each level of the pyramid, returns a pyramid
    // where each level contains the response of the classifier at the
    // corresponding level of the input pyramid.
    void predictSlidingWindow(const FeatureCollection &featPyr, vector<Mat> &responsePyr) const;

    // Print the parameters chosen for the SVM
    void printSVMParameters();

    // Get SVM weights in the shape of the original features
    //vector<float> getWeights() const;
    double getBiasTerm() const;

    // Get default parameters
    static ParametersMap getDefaultParameters();
    ParametersMap getParameters();

    //Mat renderSVMWeights(const FeatureExtractor *featExtractor);

    // Loading and saving model to file
    void load(const std::string &filename);
    void load(FILE *fp);
    void save(const std::string &filename) const;
    void save(FILE *fp) const;

    bool initialized() const { return _model != NULL; }
};

#endif // SUPPORT_VECTOR_MACHINE_H
#include "SupportVectorMachine.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

const char *SVM_TYPE        = "svm_type";
const char *KERNEL_TYPE     = "kernel_type";
const char *DEGREE          = "degree";
const char *GAMMA           = "gamma";
const char *COEF0           = "coef0";
const char *NU              = "nu";
const char *CACHE_SIZE      = "cache_size";
const char *C               = "c";
const char *EPS             = "eps";
const char *P               = "p";
const char *SHRINKING       = "shrinking";
const char *PROBABILITY     = "probability";

SupportVectorMachine::SupportVectorMachine():
    _model(NULL),
    _data(NULL)
{
    _param.nr_weight = 0;
    _param.weight_label = NULL;
    _param.weight = NULL;
}

SupportVectorMachine::SupportVectorMachine(const ParametersMap &params):
    _model(NULL),
    _data(NULL)
{
    string svm_type = params.getStr(SVM_TYPE);
    string kernel_type = params.getStr(KERNEL_TYPE);

    if(boost::iequals(svm_type,"C_SVC"))
        _param.svm_type = C_SVC;
    else if (boost::iequals(svm_type,"NU_SVC"))
        _param.svm_type = NU_SVC;
    else if (boost::iequals(svm_type,"ONE_CLASS"))
        _param.svm_type = ONE_CLASS;
    else if (boost::iequals(svm_type,"EPSILON_SVR"))
        _param.svm_type = EPSILON_SVR;
    else if (boost::iequals(svm_type,"NU_SVR"))
        _param.svm_type = NU_SVR;

    cout << "SVM constructor" << endl;

    if(boost::iequals(kernel_type,"LINEAR"))
        _param.kernel_type = LINEAR;
    else if (boost::iequals(kernel_type,"POLY"))
        _param.kernel_type = POLY;
    else if (boost::iequals(kernel_type,"RBF"))
        _param.kernel_type = RBF;
    else if (boost::iequals(kernel_type,"SIGMOID"))
        _param.kernel_type = SIGMOID;
    else if (boost::iequals(kernel_type,"PRECOMPUTED"))
        _param.kernel_type = PRECOMPUTED;

    _param.degree = params.getInt(DEGREE);
    _param.gamma = params.getFloat(GAMMA);
    _param.coef0 = params.getFloat(COEF0);
    _param.nu = params.getFloat(NU);
    _param.cache_size = params.getInt(CACHE_SIZE);
    _param.C = params.getFloat(C);
    _param.eps = params.getFloat(EPS);
    _param.p = params.getFloat(P);
    _param.shrinking = params.getInt(SHRINKING);
    _param.probability = params.getInt(PROBABILITY);

    _param.nr_weight = 0;
    _param.weight_label = NULL;
    _param.weight = NULL;
}

SupportVectorMachine::SupportVectorMachine(const std::string &modelFName):
    _model(NULL),
    _data(NULL)
{
    LOG(INFO) << "Loading svm model: " << modelFName;
    _model = load(modelFName);
    _param = _model->param;

}

void SupportVectorMachine::_deinit()
{
    if(_model != NULL) svm_free_and_destroy_model(&_model);
    _model = NULL;
}

SupportVectorMachine::~SupportVectorMachine()
{
    _deinit();
}

ParametersMap SupportVectorMachine::getDefaultParameters()
{
    ParametersMap params;
    params.set(SVM_TYPE, "C_SVC");
    params.set(KERNEL_TYPE, "LINEAR");
    params.set(DEGREE, 0);
    params.set(GAMMA, 0);
    params.set(COEF0, 0);
    params.set(NU, 0.5);
    params.set(CACHE_SIZE, 100); // In MB
    params.set(C, 0.01);
    params.set(EPS, 1e-3);
    params.set(P, 0.1);
    params.set(SHRINKING, 1);
    params.set(PROBABILITY, 0);
    return params;
}

ParametersMap SupportVectorMachine::getParameters()
{
    ParametersMap params;
    params.set(SVM_TYPE, _param.svm_type);
    params.set(KERNEL_TYPE, _param.kernel_type);
    params.set(DEGREE, _param.degree);
    params.set(GAMMA, _param.gamma);
    params.set(COEF0, _param.coef0);
    params.set(NU, _param.nu);
    params.set(CACHE_SIZE, _param.cache_size);
    params.set(C, _param.C);
    params.set(EPS, _param.eps);
    params.set(P, _param.p);
    params.set(SHRINKING, _param.shrinking);
    params.set(PROBABILITY, _param.probability);
    return params;
}

void SupportVectorMachine::printSVMParameters()
{
    cout << "SVM_TYPE: " << _param.svm_type << endl;
    cout << "KERNEL_TYPE: " << _param.kernel_type << endl;
    cout << "DEGREE: " << _param.degree << endl;
    cout << "GAMMA: " << _param.gamma << endl;
    cout << "COEF0: " << _param.coef0 << endl;
    cout << "NU: " << _param.nu << endl;
    cout << "CACHE_SIZE: " << _param.cache_size << endl;
    cout << "C: " << _param.C << endl;
    cout << "EPS: " << _param.eps << endl;
    cout << "P: " << _param.p << endl;
    cout << "SHRINKING: " << _param.shrinking << endl;
    cout << "PROBABILITY: " << _param.probability << endl;
}

void SupportVectorMachine::train(const std::vector<float> &labels, FeatureCollection &features, std::string svmModelFName)
{
     if(labels.size() != features.size()) throw std::runtime_error("ERROR: Database size is different from feature set size!");

    printSVMParameters();

    // Figure out size and number of feature vectors
    int nVecs = labels.size();
    int dim = features[0].size();

    // Allocate memory
    svm_problem problem;
    problem.l = nVecs;
    problem.y = new double[nVecs];
    problem.x = new svm_node*[nVecs];
    if(_data) delete [] _data;

    // entry to -1
    _data = new svm_node[nVecs * (dim + 1)];

    // Iterate over the feature vectors, copying the data to the appropiate data structures
    for(int k = 0; k < nVecs; k++){
        // Copy the label to problem.y
        problem.y[k] = (double) labels[k];

        // Copy the address where the k-th feature starts
        problem.x[k] = &_data[k*(dim+1)];

        // Copy the feature vector into _data
        Feature currentFeature = features[k];
        for(int i = 0; i < dim+1; i++){
            if(i != dim){
                _data[k*(dim+1)+i].index = i;
                _data[k*(dim+1)+i].value = currentFeature[i];
            }
            else{
                // Set index for the last svm_node to -1 to indicate the feature has ended
                _data[k*(dim+1)+dim].index = -1;
            }
        }
    }

    // Train the model
    if(_model != NULL) svm_free_and_destroy_model(&_model);
    _model = svm_train(&problem, &_param);

    LOG(INFO) << "Saving model file to: " << svmModelFName;
    save(svmModelFName);

    // Cleanup
    delete [] problem.y;
    delete [] problem.x;
}

float SupportVectorMachine::predict(const Feature &feature) const
{
    int dim = feature.size();

    svm_node *svmNode = new svm_node[dim + 1];

    svm_node *svmNodeIter = svmNode;

    for(int i = 0; i < dim; i++) {
        svmNodeIter->index = i;
        svmNodeIter->value = feature[i];
        svmNodeIter++;
    }
    svmNodeIter->index = -1;

    double decisionValue;
    float label = svm_predict_values(_model, svmNode, &decisionValue);

    delete [] svmNode;

    return decisionValue;
}

float SupportVectorMachine::predictLabel(const Feature &feature) const
{
    int dim = feature.size();

    svm_node *svmNode = new svm_node[dim + 1];

    svm_node *svmNodeIter = svmNode;

    for(int i = 0; i < dim; i++) {
        svmNodeIter->index = i;
        svmNodeIter->value = feature[i];
        svmNodeIter++;
    }
    svmNodeIter->index = -1;

    double decisionValue;
    cout << feature.size() << endl;
    float label = svm_predict_values(_model, svmNode, &decisionValue);
    delete [] svmNode;

    return label;
}

float SupportVectorMachine::predictLabel(const Feature &feature, double& decisionValue) const
{
    int dim = feature.size();

    svm_node *svmNode = new svm_node[dim + 1];

    svm_node *svmNodeIter = svmNode;

    for(int i = 0; i < dim; i++) {
        svmNodeIter->index = i;
        svmNodeIter->value = feature[i];
        svmNodeIter++;
    }
    svmNodeIter->index = -1;

    float label = svm_predict_values(_model, svmNode, &decisionValue);

    delete [] svmNode;

    return label;
}

std::vector<float> SupportVectorMachine::predict(const FeatureCollection &fset)
{
    //printSVMParameters();

    int n = fset.size();
    std::vector<float> preds(n);
    for(int i = 0; i < n; i++) {
        float percent;
        printf("\033[s");
        // Print progress string
        if((i+1)%1000 == 0 || (i+1) == n)
        {
            percent = ((i+1)*100)/n;
            cout << percent << "% ... ";
            fflush(stdout);
            printf("\033[u");
        }

        preds[i] = predict(fset[i]);
    }
    printf("\n");

    return preds;
}

std::vector<float> SupportVectorMachine::predictLabel(const FeatureCollection &fset) const
{
    int n = fset.size();
    std::vector<float> preds(n);
    for(int i = 0; i < n; i++) {
        float percent;
        printf("\033[s");
        // Print progress string
        if((i+1)%1000 == 0 || (i+1) == n)
        {
            percent = ((i+1)*100)/n;
            cout << percent << "% ... ";
            fflush(stdout);
            printf("\033[u");
        }

        double decisionValue;
        preds[i] = predictLabel(fset[i], decisionValue);
    }
    printf("\n");
    return preds;
}

std::vector<float> SupportVectorMachine::getDetector() const
{
    if(_model == NULL)
        throw std::runtime_error("ERROR: Asking for SVM bias term but there is no model. Either load one from file or train one before.");

    std::vector<float> weights;
    
    const double * const * sv_coef = _model->sv_coef;
    const svm_node * const *SV = _model->SV;
    int l = _model->l;
    //_model->label;

    const svm_node* p_tmp = SV[0];
    int len = 0;
    while(p_tmp->index != -1)
    {
        len++;
        p_tmp++;
    }

    weights.resize(len+1);

    for(int i = 0; i < l; i++)
    {
        double svcoef = sv_coef[0][i];
        const svm_node* p = SV[i];
        while( p->index != -1)
        {
            weights[p->index-1] += float(svcoef * p->value);
            p++;
        }
    }
    weights[len] = float(-_model->rho[0]);
    return weights;
}

double SupportVectorMachine::getBiasTerm() const
{
    if(_model == NULL)
        throw std::runtime_error("ERROR: Asking for SVM bias term but there is no model. Either load one from file or train one before.");
    return _model->rho[0];
}

svm_model * SupportVectorMachine::load(const std::string &filename)
{
    _deinit();
    // FILE *f = fopen(filename.c_str(), "rb");
    // if(f == NULL) throw "Failed to open file " + filename + " for reading";
    
    return svm_load_model(filename.c_str());
}

void SupportVectorMachine::save(const std::string &filename) const
{
    svm_save_model(filename.c_str(),_model);
}
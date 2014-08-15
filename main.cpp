#include "Common.h"
#include "Utils.h"
#include "Feature.h"
#include "SupportVectorMachine.h"
#include "PrecisionRecall.h"
#include "ObjectDetector.h"
#include "SubBandImagePyramid.h"
#include "FileIO.h"
#include "ImageDatabase.h"

using namespace std;

void printUsage(const std::string &execName)
{
    printf("Usage:\n");
    printf("\t%s -h\n", execName.c_str());
    printf("\t%s TRAIN   -p <in:params> [-c <svm C param>] <in:database> <out:svm model>\n", execName.c_str());
    printf("\t%s TRAIN   -f <feature type> [-c <svm C param>] <in:database> <out:svm model>\n", execName.c_str());
    printf("\t%s PRED    <in:database> <in:svm model> [<out:prcurve.pr>] [<out:database.preds>]\n", execName.c_str());
    printf("\t%s PREDSL  [-p <in:params>] <in:database> <in:svm model> [<out:prcurve.pr>] [<out:database.preds>]\n", execName.c_str());
}

void parseCommandLineOptions(int argc, char **argv, vector<std::string> &args, map<std::string, string> &opts)
{
    for (int i = 0; i < argc; i++) {
        if(strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            exit(EXIT_SUCCESS);
        }

        if(argv[i][0] == '-') {
            opts[argv[i]] = argv[i + 1];
            i++;
        } else {
            args.push_back(argv[i]);
        }
    }
}

int mainSVMTrain(const vector<string> &args, const map<string, string> &opts)
{
    if(args.size() < 3) {
        throw CError("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    string dbFName = args[2];
    string svmModelFName = args[3];

    PascalImageDatabase db(dbFName.c_str());
    cout << db << endl;

    ParametersMap featParams;
    if(opts.count("-f") == 1) {
        string featureType = opts.at("-f");
        featParams = FeatureExtractor::getDefaultParameters(featureType);
    } else if(opts.count("-p") == 1) {
        string paramsFName = opts.at("-p");

        map<string, ParametersMap> allParams;
        loadFromFile(paramsFName, allParams);

        if(allParams.count(FEATURE_EXTRACTOR_KEY)) {
            LOG(INFO) << "Using feature extractor paramaters from file ";
            featParams = allParams[FEATURE_EXTRACTOR_KEY];
        } else {
            LOG(INFO) << "Using default parameters for feature extractor";
        }
    } else {
        throw CError("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    double svmC = 0.01;
    if(opts.count("-c") == 1) svmC = atof(opts.at("-c").c_str());

    FeatureExtractor *featExtractor = FeatureExtractor::create(featParams);

    LOG(INFO) << "Feature type: " << featExtractor->getFeatureType();

    LOG(INFO) << "Extracting features";
    FeatureCollection features;
    (*featExtractor)(db, features);

    LOG(INFO) << "Training SVM";
    SupportVectorMachine svm;

    svm.train(db.getLabels(), features, svmC);

    saveToFile(svmModelFName, svm, featExtractor);

    delete featExtractor;
    return EXIT_SUCCESS;
}

int mainSVMPredict(const vector<string> &args, const map<string, string> &opts)
{
    if(args.size() < 3 || args.size() > 6) {
        throw CError("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    string dbFName = args[2];
    string svmModelFName = args[3];
    string prFName = (args.size() >= 5) ? args[4] : "";
    string predsFName = (args.size() >= 6) ? args[5] : "";

    PascalImageDatabase db(dbFName.c_str());
    cout << db << endl;

    LOG(INFO) << "Loading SVM model and feature extractor from file";
    SupportVectorMachine svm;
    FeatureExtractor *featExtractor = NULL;
    loadFromFile(svmModelFName, svm, &featExtractor);

    LOG(INFO) << "Extracting features";
    FeatureCollection features;
    (*featExtractor)(db, features);

    LOG(INFO) << "Predicting";
    vector<float> preds = svm.predict(features);

    LOG(INFO) << "Computing Precision Recall Curve";
    PrecisionRecall pr(db.getLabels(), preds);
    LOG(INFO) << "Average precision: " << pr.getAveragePrecision();

    if(prFName.size() != 0) pr.save(prFName.c_str());
    if(predsFName.size() != 0) {
        PascalImageDatabase predsDb(preds, db.getFilenames());
        predsDb.save(predsFName.c_str());
    }

    return EXIT_SUCCESS;
}

int mainSVMPredictSlidingWindow(const vector<string> &args, const map<string, string> &opts)
{
    // Detection over multiple scales with non maxima suppression
    if(args.size() < 4) {
        throw CError("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    string dbFName = args[2];
    string svmModelFName = args[3];
    string prFName = (args.size() >= 5) ? args[4] : "";
    string predsFName = (args.size() >= 6) ? args[5] : "";

    //ParametersMap imPyrParams = SBFloatPyramid::getDefaultParameters();
    ParametersMap obDetParams = ObjectDetector::getDefaultParameters();
    if(opts.count("-p") == 1) {
        string paramsFName = opts.at("-p");

        map<string, ParametersMap> allParams;
        loadFromFile(paramsFName, allParams);

        // if(allParams.count(IMAGE_PYRAMID_KEY)) {
        //     LOG(INFO) << "Using image pyarmid paramaters from file ";
        //     imPyrParams = allParams[IMAGE_PYRAMID_KEY];
        // } else {
        //     LOG(INFO) << "Using default parameters for image pyaramid";
        // }

        if(allParams.count(OBJECT_DETECTOR_KEY)) {
            LOG(INFO) << "Using NMS paramaters from file ";
            obDetParams = allParams[OBJECT_DETECTOR_KEY];
        } else {
            LOG(INFO) << "Using default parameters for NMS";
        }
    }

    LOG(INFO) << "Loading image database";
    ImageDatabase db(dbFName);

    LOG(INFO) << "Loading SVM model and features extractor from file";
    SupportVectorMachine svm;
    FeatureExtractor *featExtractor = NULL;
    loadFromFile(svmModelFName, svm, &featExtractor);

    LOG(INFO) << "Initializing object detector";
    ObjectDetector obdet(obDetParams);

    vector<vector<Detection> > dets(db.getSize());
    for(int i = 0; i < db.getSize(); i++) {
        LOG(INFO) << "Processing image " << setw(4) << (i + 1) << " of " << db.getSize();

        string imgFName = db.getFilenames()[i];

        // // load image
        // CByteImage img;
        // ReadFile(img, imgFName.c_str());

        // // build pyramid
        // CFloatImage imgF(img.Shape());
        // TypeConvert(img, imgF);
        // SBFloatPyramid imgPyr(imgF, imPyrParams);

        // // Extracting features on image pyramid
        // FeaturePyramid featPyr;
        // (*featExtractor)(imgPyr, featPyr);

        // // Computing SVM response for every level
        // SBFloatPyramid responsePyr;
        // svm.predictSlidingWindow(featPyr, responsePyr);

        // // Extracting detections from response pyramid
        // obdet(responsePyr, svm.getROISize(), featExtractor->scaleFactor(), dets[i]);
    }

    ImageDatabase predsDb(dets, db.getFilenames());

    LOG(INFO) << "Computing Precision Recall Curve";
    vector<float> labels, response;
    int nGroundTruthDetections;
    computeLabels(db.getDetections(), predsDb.getDetections(), labels, response, nGroundTruthDetections);

    LOG(INFO) << "Computing Precision Recall Curve";
    PrecisionRecall pr(labels, response, nGroundTruthDetections);
    LOG(INFO) << "Average precision: " << pr.getAveragePrecision();

    if(predsFName.size()) predsDb.save(predsFName.c_str());
    if(prFName.size()) pr.save(prFName.c_str());

    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    FLAGS_logtostderr = true;
    FLAGS_stderrthreshold = 0;
    google::InitGoogleLogging(argv[0]);

    vector<string> args;
    map<string, string> opts;
    parseCommandLineOptions(argc, argv, args, opts);

    try {
        if (strcasecmp(args[1].c_str(), "TRAIN") == 0) {
            return mainSVMTrain(args, opts);
        } else if (strcasecmp(args[1].c_str(), "PRED") == 0) {
            return mainSVMPredict(args, opts);
        // } else if (strcasecmp(args[1].c_str(), "PREDSL") == 0) {
        //     return mainSVMPredictSlidingWindow(args, opts);
        } else {
            printUsage(args[0]);
            return EXIT_FAILURE;
        }

    } catch(CError err) {
        LOG(ERROR) << "ERROR: Uncought exception: " << err.message << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


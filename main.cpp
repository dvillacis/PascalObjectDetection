#include "Common.h"
#include "PascalAnnotation.h"
#include "PascalImageDatabase.h"
#include "ImageDatabase.h"
#include "Feature.h"
#include "SupportVectorMachine.h"
#include "PrecisionRecall.h"
#include "ObjectDetector.h"
#include "FileIO.h"
#include "PrincipalComponentAnalysis.h"


using namespace std;

void printUsage(const std::string &execName)
{
    printf("Usage:\n");
    printf("\t%s -h\n", execName.c_str());
    printf("\t%s TRAIN   -c <category name> [-p <svm C param>] <in:database> <out:svm model>\n", execName.c_str());
    printf("\t%s PRED    -c <category name> <in:database> <in:svm model> [<out:prcurve.pr>] [<out:database.preds>]\n", execName.c_str());
    printf("\t%s PREDSL  -c <category name> <in:database> <in:svm model> [<out:prcurve.pr>] [<out:database.preds>]\n", execName.c_str());
    printf("\t%s PCA     -c <category name> <in:database> [<out:pca_data.dat>]", execName.c_str());
    printf("\t%s VIDEO   <in:svm model>", execName.c_str());
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
        throw "ERROR: Incorrect number of arguments. Run command with flag -h for help.";
    }

    double t = (double)getTickCount();

    string dbFName = args[2];
    string svmModelFName = args[3];
    string category;

    LOG(INFO) << "Creating feature extractor";
    ParametersMap featParams;
    if(opts.count("-c") == 1) {
        category = opts.at("-c");
        featParams = FeatureExtractor::getDefaultParameters("hog");
    } else {
        throw "ERROR: Incorrect number of arguments. Run command with flag -h for help.";
    }
    
    LOG(INFO) << "Creating the image database";
    PascalImageDatabase db(dbFName.c_str(), category);
    cout << db << endl;

    FeatureExtractor *featExtractor = FeatureExtractor::create(featParams);

    LOG(INFO) << "Category: " << category;

    LOG(INFO) << "Extracting features";
    FeatureCollection features;
    (*featExtractor)(db, features);

    LOG(INFO) << "Obtaining svm parameters";
    ParametersMap svmParams;
    if(opts.count("-p") == 1){
        string paramsSVMFName = opts.at("-p");
        map<string, ParametersMap> allParams;
        loadFromFile(paramsSVMFName, allParams);
        if(allParams.count(SVM_CONFIG_KEY)){
            LOG(INFO) << "Using svm parameters from file: " << paramsSVMFName;
            svmParams = allParams[SVM_CONFIG_KEY];
        } else {
            throw "ERROR: Problem obtaining the parameters from file: " + paramsSVMFName;
        }
    } else {
        LOG(INFO) << "Using default svm parameters";
        svmParams = SupportVectorMachine::getDefaultParameters();
    }

    LOG(INFO) << "Training SVM";
    SupportVectorMachine svm(svmParams);
    svm.train(db.getLabels(), features);

    saveToFile(svmModelFName, svm);

    delete featExtractor;

    t = (double)getTickCount() - t;
    LOG(INFO) << "Training completed in " << t/getTickFrequency() << " seconds.";

    return EXIT_SUCCESS;
}

int mainSVMPredict(const vector<string> &args, const map<string, string> &opts)
{
    // TODO: Fix the parameters input

    if(args.size() < 4 || args.size() > 6) {
        throw "ERROR: Incorrect number of arguments. Run command with flag -h for help.";
    }

    double t = (double)getTickCount();

    string dbFName = args[2];
    string svmModelFName = args[3];
    string prFName = (args.size() >= 5) ? args[4] : "";
    string predsFName = (args.size() >= 6) ? args[5] : "";
    string predsFName_label = predsFName+"_target";
    string predsFName_scores = predsFName+"_scores";

    string category;
    if(opts.count("-c") == 1) {
        category = opts.at("-c");
    } else {
        throw "ERROR: Incorrect number of arguments. Run command with flag -h for help.";
    }

    LOG(INFO) << "Creating the image database";
    PascalImageDatabase db(dbFName.c_str(), category);
    cout << db << endl;

    LOG(INFO) << "Loading SVM model and feature extractor from file";
    SupportVectorMachine svm(svmModelFName);
    FeatureExtractor *featExtractor = FeatureExtractor::create(FeatureExtractor::getDefaultParameters("hog"));
    //loadFromFile(svmModelFName, svm);

    LOG(INFO) << "Extracting features";
    FeatureCollection features;
    (*featExtractor)(db, features);

    LOG(INFO) << "Predicting";
    vector<float> preds = svm.predict(features);
    vector<float> predLabels = svm.predictLabel(features);

    LOG(INFO) << "Computing Precision Recall Curve";
    PrecisionRecall pr(db.getLabels(), preds);
    LOG(INFO) << "Average precision: " << pr.getAveragePrecision();

    if(prFName.size() != 0) pr.save(prFName.c_str());
    if(predsFName.size() != 0) {
        PascalImageDatabase predsDb(predLabels, db.getFilenames());
        predsDb.save(predsFName.c_str());
        PascalImageDatabase targetDb(db.getLabels(), db.getFilenames());
        targetDb.save(predsFName_label.c_str());
        PascalImageDatabase scoresDb(preds, db.getFilenames());
        scoresDb.save(predsFName_scores.c_str());
    }

    LOG(INFO) << "Cross Validation completed in " << t/getTickFrequency() << " seconds.";

    return EXIT_SUCCESS;
}

int mainSVMPredictSlidingWindow(const vector<string> &args, const map<string, string> &opts)
{
    // TODO: Fix parameter input

    // Detection over multiple scales with non maxima suppression
    if(args.size() < 5) {
        throw "ERROR: Incorrect number of arguments. Run command with flag -h for help.";
    }

    string dbFName = args[2];
    string svmModelFName = args[3];
    string prFName = (args.size() >= 5) ? args[4] : "";
    string predsFName = (args.size() >= 6) ? args[5] : "";

    //ParametersMap imPyrParams = SBFloatPyramid::getDefaultParameters();
    // ParametersMap obDetParams = ObjectDetector::getDefaultParameters();
    // if(opts.count("-p") == 1) {
    //     string paramsFName = opts.at("-p");

    //     map<string, ParametersMap> allParams;
    //     loadFromFile(paramsFName, allParams);

    //     // if(allParams.count(IMAGE_PYRAMID_KEY)) {
    //     //     LOG(INFO) << "Using image pyarmid paramaters from file ";
    //     //     imPyrParams = allParams[IMAGE_PYRAMID_KEY];
    //     // } else {
    //     //     LOG(INFO) << "Using default parameters for image pyaramid";
    //     // }

    //     if(allParams.count(OBJECT_DETECTOR_KEY)) {
    //         LOG(INFO) << "Using NMS paramaters from file: " << paramsFName;
    //         obDetParams = allParams[OBJECT_DETECTOR_KEY];
    //     } else {
    //         LOG(INFO) << "Using default parameters for NMS";
    //     }
    // }

    string category;
    if(opts.count("-c") == 1) {
        category = opts.at("-c");
    } else {
        throw "ERROR: Incorrect number of arguments. Run command with flag -h for help.";
    }

    LOG(INFO) << "Loading image database";
    ImageDatabase db(dbFName, category);
    cout << db << endl;

    LOG(INFO) << "Loading SVM model and features extractor from file";
    SupportVectorMachine svm;
    HOGFeatureExtractor *featExtractor = new HOGFeatureExtractor();
    loadFromFile(svmModelFName, svm);

    LOG(INFO) << "Initializing object detector";
    ObjectDetector obdet(svm);
    //obdet.setDetector(svm.getDetector());

    vector<vector<Detection> > dets(db.getSize());

    for(int i = 0; i < db.getSize(); i++) {
        LOG(INFO) << "Processing image " << setw(4) << (i + 1) << " of " << db.getSize();

        string imgFName = db.getFilenames()[i];

        // load image
        Mat img = imread(imgFName,CV_LOAD_IMAGE_COLOR);

        // Extracting detections from the source image
        LOG(INFO) << " --> Extracting detections from the source image";
        vector<Rect> found;
        obdet.getDetections(img, found);

        for(int j = 0; j < found.size(); j++){
            rectangle(img,found[j].tl(),found[j].br(),Scalar(255,0,0),2);
        }

        imshow("Custom Detection", img);
        waitKey(0);

        img.release();

        // build pyramid
        // LOG(INFO) << " --> Building pyramid";
        // vector<Mat> imgPyr;
        // buildPyramid(img, imgPyr, 2);

        // Extracting features on image pyramid
        // LOG(INFO) << " --> Extracting features";
        // FeatureCollection featPyr;
        // (*featExtractor)(imgPyr, featPyr);

        // Computing SVM response for every level
        // LOG(INFO) << " --> Computing SVM response";
        // vector<Mat> responsePyr;
        // // SBFloatPyramid responsePyr;
        // svm.predictSlidingWindow(featPyr, responsePyr);

        
    }

    // ImageDatabase predsDb(dets, db.getFilenames());

    // LOG(INFO) << "Computing Precision Recall Curve";
    // vector<float> labels, response;
    // int nGroundTruthDetections;
    // computeLabels(db.getDetections(), predsDb.getDetections(), labels, response, nGroundTruthDetections);

    // LOG(INFO) << "Computing Precision Recall Curve";
    // PrecisionRecall pr(labels, response, nGroundTruthDetections);
    // LOG(INFO) << "Average precision: " << pr.getAveragePrecision();

    // if(predsFName.size()) predsDb.save(predsFName.c_str());
    // if(prFName.size()) pr.save(prFName.c_str());

    return EXIT_SUCCESS;
}

int mainPCA(const vector<string> &args, const map<string, string> &opts)
{
    if(args.size() < 4) {
        throw "ERROR: Incorrect number of arguments. Run command with flag -h for help.";
    }

    double t = (double)getTickCount();

    LOG(INFO) << "Performing a PCA analysis on the images database";

    string dbFName = args[2];
    string pcaFName = args[3];
    string category;

    LOG(INFO) << "Obtaining feature extractor parameters";
    ParametersMap featParams;
    if(opts.count("-c") == 1) {
        category = opts.at("-c");
        featParams = FeatureExtractor::getDefaultParameters("hog");
    } else {
        throw "ERROR: Incorrect number of arguments. Run command with flag -h for help.";
    }

    LOG(INFO) << "Creating the image database";
    PascalImageDatabase db(dbFName.c_str(), category);
    cout << db << endl;

    FeatureExtractor *featExtractor = FeatureExtractor::create(featParams);

    LOG(INFO) << "Category: " << category;

    LOG(INFO) << "Extracting HOG features";
    FeatureCollection features;
    (*featExtractor)(db, features);

    LOG(INFO) << "Performing PCA on the obtained HOG features";
    int num_samples = features.size();
    int num_features = features[0].size();
    Mat data(num_features,num_samples,CV_32FC1,Scalar(0));
    PrincipalComponentAnalysis pca;
    pca.pre_process(features,data);
    pca.compute(data,db);
    pca.savePCAFile(pcaFName);

    LOG(INFO) << "PCA completed in " << t/getTickFrequency() << " seconds.";

    return EXIT_SUCCESS;
}

int mainVIDEO(const vector<string> &args, const map<string, string> &opts)
{
    string svmModelFName = args[2];

    LOG(INFO) << "Loading SVM model and features extractor from file";
    SupportVectorMachine svm;
    HOGFeatureExtractor *featExtractor = new HOGFeatureExtractor();
    loadFromFile(svmModelFName, svm);

    LOG(INFO) << "Initializing object detector";
    //vector<float> svmDetector = svm.getDetector();
    ObjectDetector obdet(svm);
    //obdet.setDetector(svm.getDetector());

    VideoCapture capture(0);
    if(!capture.isOpened())
        throw "Couldnt open the camera";
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 256);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 256);
    string windowName = "Live Feed Detection";
    for(;;)
    {
        Mat frame;
        vector<Rect> found;
        capture >> frame;
        obdet.getDetections(frame,found);
        for(int j = 0; j < found.size(); j++){
            rectangle(frame,found[j].tl(),found[j].br(),Scalar(255,0,0),2);
        }
        imshow(windowName,frame);
    }

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
        } else if (strcasecmp(args[1].c_str(), "PREDSL") == 0) {
            return mainSVMPredictSlidingWindow(args, opts);
        } else if (strcasecmp(args[1].c_str(), "PCA") == 0) {
            return mainPCA(args,opts);
        } else if (strcasecmp(args[1].c_str(), "VIDEO") == 0) {
            return mainVIDEO(args,opts);
        } else {
            printUsage(args[0]);
            return EXIT_FAILURE;
        }

    } catch(std::exception err) {
        LOG(ERROR) << "ERROR: Uncought exception: " << err.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


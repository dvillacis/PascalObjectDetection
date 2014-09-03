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
    printf("\t%s TRAIN      -c <category name> [-p <svm C param>] <in:database> <out:svm model>\n", execName.c_str());
    printf("\t%s VAL        -c <category name> <in:database> <in:svm model> [<out:prcurve.pr>] [<out:database.preds>]\n", execName.c_str());
    printf("\t%s TEST       -c <category name> <in:database> <in:svm model> [<out:prcurve.pr>] [<out:database.preds>]\n", execName.c_str());
    printf("\t%s PCA        -c <category name> <in:database> [<out:pca_data.dat>]\n", execName.c_str());
    printf("\t%s DEMO       -c <category name> <in:database> <in:svm model>\n\n", execName.c_str());
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
    if(args.size() != 4) {
        throw std::runtime_error("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    double t = (double)getTickCount();

    string dbFName = args[2];
    string svmModelFName = args[3];
    string category;
    if(opts.count("-c") == 1) {
        category = opts.at("-c");
    } else {
        throw std::runtime_error("ERROR: Category not specified. Run command with flag -h for help.");
    }

    LOG(INFO) << "Obtaining svm parameters";
    ParametersMap svmParams;
    if(opts.count("-p") == 1){
        string paramsSVMFName = opts.at("-p");
        if(boost::filesystem::exists(paramsSVMFName))
        {
            map<string, ParametersMap> allParams;
            loadFromFile(paramsSVMFName, allParams);
            if(allParams.count(SVM_CONFIG_KEY)){
                LOG(INFO) << "Using svm parameters from file: " << paramsSVMFName;
                svmParams = allParams[SVM_CONFIG_KEY];
            } else {
                throw std::runtime_error("ERROR: Problem obtaining the parameters from file: " + paramsSVMFName);
            }
        } else {
            throw std::runtime_error("ERROR: SVM configuration file doesn't exist in: " + paramsSVMFName);
        }
    } else {
        LOG(INFO) << "Using default svm parameters";
        svmParams = SupportVectorMachine::getDefaultParameters();
    }

    
    LOG(INFO) << "Creating the image database";
    if(boost::filesystem::exists(dbFName)) {

        PascalImageDatabase db(dbFName.c_str(), category);
        cout << db << endl;

        LOG(INFO) << "Creating feature extractor";
        ParametersMap featParams;
        featParams = FeatureExtractor::getDefaultParameters("hog");
        FeatureExtractor *featExtractor = FeatureExtractor::create(featParams);

        LOG(INFO) << "Category: " << category;

        LOG(INFO) << "Extracting features";
        FeatureCollection features;
        (*featExtractor)(db, features);

        LOG(INFO) << "Scaling the feature vector";
        FeatureCollection scaledFeatures;
        featExtractor->scale(features,scaledFeatures);

        // Remove features from memory
        FeatureCollection().swap(features);

        LOG(INFO) << "Training SVM";
        SupportVectorMachine svm(svmParams);
        svm.train(db.getLabels(), scaledFeatures, svmModelFName);

        //saveToFile(svmModelFName, svm);
        LOG(INFO) << "SVM Model saved in: " << svmModelFName;

        delete featExtractor;

        t = (double)getTickCount() - t;
        LOG(INFO) << "Training completed in " << t/getTickFrequency() << " seconds.";

    } else {
        throw std::runtime_error("ERROR: Pascal database training file doesn't exist in: " + dbFName);
    }

    return EXIT_SUCCESS;
}

int mainSVMVal(const vector<string> &args, const map<string, string> &opts)
{
    if(args.size() != 6) {
        throw std::runtime_error("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
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
        throw std::runtime_error("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    if(boost::filesystem::exists(dbFName))
    {
        if(boost::filesystem::exists(svmModelFName))
        {

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

            LOG(INFO) << "Scaling the feature vector";
            FeatureCollection scaledFeatures;
            featExtractor->scale(features,scaledFeatures);

            // Remove features from memory
            FeatureCollection().swap(features);

            LOG(INFO) << "Predicting";
            vector<float> preds = svm.predict(scaledFeatures);
            //vector<float> predLabels = svm.predictLabel(features);

            LOG(INFO) << "Computing Precision Recall Curve";
            PrecisionRecall pr(db.getLabels(), preds);
            LOG(INFO) << "Average precision: " << pr.getAveragePrecision();

            if(prFName.size() != 0) pr.save(prFName.c_str());
            if(predsFName.size() != 0) {
                PascalImageDatabase predsDb(preds, db.getFilenames());
                predsDb.save(predsFName.c_str());
                // PascalImageDatabase targetDb(db.getLabels(), db.getFilenames());
                // targetDb.save(predsFName_label.c_str());
                // PascalImageDatabase scoresDb(preds, db.getFilenames());
                // scoresDb.save(predsFName_scores.c_str());
            }

            delete featExtractor;

            t = (double)getTickCount() - t;
            LOG(INFO) << "Cross Validation completed in " << t/getTickFrequency() << " seconds.";

            return EXIT_SUCCESS;
        }
        else
        {
            throw std::runtime_error("ERROR: SVM Model file doesn't exist in: " + svmModelFName);
        }
    }
    else
    {
        throw std::runtime_error("ERROR: Pascal cross validation database file doesn't exist in: " + dbFName);
    }
}

int mainSVMTest(const vector<string> &args, const map<string, string> &opts)
{
    // Detection over multiple scales with non maxima suppression
    if(args.size() != 6) {
        throw std::runtime_error("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    string dbFName = args[2];
    string svmModelFName = args[3];
    string prFName = (args.size() >= 5) ? args[4] : "";
    string predsFName = (args.size() >= 6) ? args[5] : "";

    string category;
    if(opts.count("-c") == 1) {
        category = opts.at("-c");
    } else {
        throw std::runtime_error("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    if(boost::filesystem::exists(dbFName))
    {
        if(boost::filesystem::exists(svmModelFName))
        {

            LOG(INFO) << "Loading image database";
            ImageDatabase db(dbFName, category);
            cout << db << endl;

            LOG(INFO) << "Loading SVM model and features extractor from file";
            SupportVectorMachine svm(svmModelFName);
            FeatureExtractor *featExtractor = FeatureExtractor::create(FeatureExtractor::getDefaultParameters("hog"));
            //loadFromFile(svmModelFName, svm);

            LOG(INFO) << "Initializing object detector";
            ObjectDetector obdet(svm);

            vector<vector<Detection> > dets(db.getSize());

            for(int i = 0; i < db.getSize(); i++) {
                LOG(INFO) << "Processing image " << setw(4) << (i + 1) << " of " << db.getSize();

                string imgFName = db.getFilenames()[i];

                // load image
                Mat img = imread(imgFName,CV_LOAD_IMAGE_COLOR);

                // Extracting detections from the source image
                LOG(INFO) << " --> Extracting detections from the source image";
                vector<Detection> found;
                obdet.getDetections(img, found);
                dets[i] = found;

                img.release();
                
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

            delete featExtractor;

            return EXIT_SUCCESS;
        }
        else
        {
            throw std::runtime_error("ERROR: SVM Model file doesn't exist in: " + svmModelFName);
        }
    }
    else
    {
        throw std::runtime_error("ERROR: Pascal testing database file doesn't exist in: " + dbFName);
    }
}

int mainPCA(const vector<string> &args, const map<string, string> &opts)
{
    if(args.size() < 4) {
        throw std::runtime_error("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
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
        throw std::runtime_error("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    if(boost::filesystem::exists(dbFName))
    {
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

        t = (double)getTickCount() - t;
        LOG(INFO) << "PCA completed in " << t/getTickFrequency() << " seconds.";

        return EXIT_SUCCESS;
    }
    else
    {
        throw std::runtime_error("ERROR: Pascal database file doesn't exist in: " + dbFName);
    }
}

int mainDEMO(const vector<string> &args, const map<string, string> &opts)
{
    if(args.size() != 4) {
        throw std::runtime_error("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    string dbFName = args[2];
    string svmModelFName = args[3];
    string prFName = (args.size() >= 5) ? args[4] : "";
    string predsFName = (args.size() >= 6) ? args[5] : "";

    string category;
    if(opts.count("-c") == 1) {
        category = opts.at("-c");
    } else {
        throw std::runtime_error("ERROR: Incorrect number of arguments. Run command with flag -h for help.");
    }

    if(boost::filesystem::exists(dbFName))
    {
        if(boost::filesystem::exists(svmModelFName))
        {

            LOG(INFO) << "Loading image database";
            ImageDatabase db(dbFName, category);
            cout << db << endl;

            LOG(INFO) << "Loading SVM model and features extractor from file";
            SupportVectorMachine svm(svmModelFName);
            FeatureExtractor *featExtractor = FeatureExtractor::create(FeatureExtractor::getDefaultParameters("hog"));
            //loadFromFile(svmModelFName, svm);

            LOG(INFO) << "Initializing object detector";
            ObjectDetector obdet(svm);

            vector<vector<Detection> > dets(db.getSize());

            for(int i = 0; i < db.getSize(); i++) {
                LOG(INFO) << "Processing image " << setw(4) << (i + 1) << " of " << db.getSize();

                string imgFName = db.getFilenames()[i];

                // load image
                Mat img = imread(imgFName,CV_LOAD_IMAGE_COLOR);

                // Extracting detections from the source image
                LOG(INFO) << " --> Extracting detections from the source image";
                vector<Detection> found;
                obdet.getDetections(img, found);

                drawDetections(img,found);

                imshow("Custom Detection", img);
                waitKey(0);

                img.release();
                
            }
            delete featExtractor;
            return EXIT_SUCCESS;
        }
        else
        {
            throw std::runtime_error("ERROR: SVM Model file doesn't exist in: " + svmModelFName);
        }
    }
    else
    {
        throw std::runtime_error("ERROR: Pascal testing database file doesn't exist in: " + dbFName);
    }
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
        if(args.size() < 2) {
            printUsage(args[0]);
            return EXIT_FAILURE;
        }
        if (strcasecmp(args[1].c_str(), "TRAIN") == 0) {
            return mainSVMTrain(args, opts);
        } else if (strcasecmp(args[1].c_str(), "VAL") == 0) {
            return mainSVMVal(args, opts);
        } else if (strcasecmp(args[1].c_str(), "TEST") == 0) {
            return mainSVMTest(args, opts);
        } else if (strcasecmp(args[1].c_str(), "PCA") == 0) {
            return mainPCA(args,opts);
        } else if (strcasecmp(args[1].c_str(), "DEMO") == 0) {
            return mainDEMO(args,opts);
        } else {
            printUsage(args[0]);
            return EXIT_FAILURE;
        }

    } catch(std::exception& err) {
        LOG(ERROR) << err.what();
        LOG(ERROR) << "Quitting ...";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


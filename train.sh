#! /bin/bash
clear

echo "This script automates the training and cross validation for the Object Detection Project"

echo "C = 0.01 GAMMA = 0.001"

SVM_MODEL_DIR=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_001-c0_0/svmModel.dat
TRAIN_DATABASE_DIR=/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/person_train.txt
CV_DATABASE_DIR=/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/person_trainval.txt
PREDS_DIR=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_001-c0_0/prcurve.pr
PREDS_DATABASE_DIR=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_001-c0_0/database.preds
SVM_CONFIG=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_001-c0_0/svm.config

echo "Starting the training ... "

./objdet TRAIN -c person -p $SVM_CONFIG $TRAIN_DATABASE_DIR $SVM_MODEL_DIR

# echo "Staring the cross validation ..."

# ./objdet PRED -f PERSON $CV_DATABASE_DIR $SVM_MODEL_DIR $PREDS_DIR $PREDS_DATABASE_DIR

echo "C = 0.01 GAMMA = 0.01"

SVM_MODEL_DIR=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_01-c0_0/svmModel.dat
TRAIN_DATABASE_DIR=/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/person_train.txt
CV_DATABASE_DIR=/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/person_trainval.txt
PREDS_DIR=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_01-c0_0/prcurve.pr
PREDS_DATABASE_DIR=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_01-c0_0/database.preds
SVM_CONFIG=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_01-c0_0/svm.config

echo "Starting the training ... "

./objdet TRAIN -c person -p $SVM_CONFIG $TRAIN_DATABASE_DIR $SVM_MODEL_DIR

# echo "Staring the cross validation ..."

# ./objdet PRED -f PERSON $CV_DATABASE_DIR $SVM_MODEL_DIR $PREDS_DIR $PREDS_DATABASE_DIR

echo "C = 0.01 GAMMA = 0.1"

SVM_MODEL_DIR=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_1-c0_0/svmModel.dat
TRAIN_DATABASE_DIR=/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/person_train.txt
CV_DATABASE_DIR=/Users/david/Documents/Development/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/person_trainval.txt
PREDS_DIR=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_1-c0_0/prcurve.pr
PREDS_DATABASE_DIR=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_1-c0_0/database.preds
SVM_CONFIG=/Volumes/EXTERNAL/DISSERTATION/MODELS/PASCAL/PERSON/HOG-SIGMOID_SVM/C_0_01-G_0_1-c0_0/svm.config

echo "Starting the training ... "

./objdet TRAIN -c person -p $SVM_CONFIG $TRAIN_DATABASE_DIR $SVM_MODEL_DIR


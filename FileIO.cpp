#include "FileIO.h"

void saveToFile(const std::string &filename, const SupportVectorMachine &svm)
{
    FILE *f = fopen(filename.c_str(), "wb");
    if(f == NULL) {
        throw "Could not open file " + filename + " for writing";
    }

    svm.save(f);
    fclose(f);
}

void loadFromFile(const std::string &filename, SupportVectorMachine &svm)
{
    FILE *f = fopen(filename.c_str(), "rb");
    if(f == NULL) {
        throw "Could not open file " + filename + " for reading";
    }

    svm.load(f);

    fclose(f);
}

void saveToFile(const std::string &filename, const std::vector<Detection> &dets)
{
    std::ofstream f(filename.c_str());

    f << "detections\n";
    f << dets.size() << "\n";
    for (int i = 0; i < dets.size(); i++) {
        f << dets[i] << "\n";
    }
}

#include "PrincipalComponentAnalysis.h"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
using namespace boost::accumulators;
using namespace std;
using namespace cv;

PrincipalComponentAnalysis::PrincipalComponentAnalysis()
{

}

void PrincipalComponentAnalysis::pre_process(const FeatureCollection fset, Mat& data)
{
	int num_samples = fset.size();
	int num_features = fset[0].size();
	for(int i = 0; i < num_samples; ++i){
		Feature feat = fset[i];
		
		accumulator_set<float, stats<tag::mean, tag::moment<2> > > acc;
		for(int k = 0; k < num_features; ++k)
			acc(feat[k]);

		float mu = boost::accumulators::mean(acc);
		float std = sqrt(moment<2>(acc));

		for(int j = 0; j < num_features; ++j){
			data.at<float>(j,i) = (feat[j]-mu)/std;
		}
	}
}

void PrincipalComponentAnalysis::compute(const Mat data, const PascalImageDatabase db)
{
	int num_principal_components = 2;

	PCA pca(data, Mat(), CV_PCA_DATA_AS_COL, num_principal_components);

	for(int i = 0; i < data.cols; ++i)
	{
		Mat proj = pca.project(data.col(i));
		int label = db.getLabel(i);
		_projectedPoints.push_back(proj);
		_labels.push_back(label);
		proj.release();
	}

	//TODO: Print the % of variability retained

	LOG(INFO) << "Percentage of variability retained: ";
}

void PrincipalComponentAnalysis::savePCAFile(const string pcaFilename)
{
	ofstream f(pcaFilename.c_str());
    for (int i = 0; i < _projectedPoints.size(); ++i) {
        f << _labels[i] << " " << _projectedPoints[i].at<float>(0,0) << " " << _projectedPoints[i].at<float>(0,1) << "\n";
    }
    f.close();

    LOG(INFO) << "Output generated in file: " << pcaFilename;
}
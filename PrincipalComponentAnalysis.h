#ifndef PRINCIPLA_COMPONENT_ANALYSIS_H
#define PRINCIPLA_COMPONENT_ANALYSIS_H

#include "Common.h"
#include "Feature.h"
#include "PascalImageDatabase.h"

//! Principal Component Analysis Class
/*!
    This class reads an image database and extract the hog features for each images,
    then it performs a principal component analysis and projects the data points to the 
    2 principal eigenvectors and generates a file to be plotted.
*/

class PrincipalComponentAnalysis 
{
public:
	//! Constructor
	PrincipalComponentAnalysis();

	//! Create the normalized data matrix
	/*!
		\param fset a vector of features extracted from the database.
		\param data normalized matrix with mean = 0 and std = 1 for each sample.
	*/
	void pre_process(const FeatureCollection fset, Mat& data);

	//! Perform PCA Analysis on the normalized data matrix
	/*!
		\param data normalized matrix with mean = 0 and std = 1 for each sample.
		\param image database.
	*/
	void compute(const Mat data, const PascalImageDatabase db);

	//! Save the projected points into a text file
	/*!
		\param pcaFilename path were the pca file will be created.
	*/
	void savePCAFile(const string pcaFilename);

private:

	vector<Mat> _projectedPoints; //! List of projected points
	vector<int> _labels; //! Labels of the projected points (positive or negative sample)

};

#endif // PRINCIPLA_COMPONENT_ANALYSIS_H
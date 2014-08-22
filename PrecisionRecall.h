#ifndef PRECISION_RECALL_H
#define PRECISION_RECALL_H

#include "Common.h"

//! A single point in the Precision Recall curve
typedef struct {
	float precision, recall, threshold;
} PecisionRecallPoint;

//! Precision Recall Class
/*!
    This class computes and stores a precision recall curve
*/
class PrecisionRecall
{
private:
	//! Collected data from the predictions
	std::vector<PecisionRecallPoint> _data;

	//! Average precision calculated for the current curve
	float _averagePrecision;

public:
	//! Constructor
	/*! 
		Computes precision recall given a set of gound truth labels in gt and
	 	a set of predictions made by our classifier in preds.
	 	\param gt Vector of ground truth detections
	 	\param preds Vector of svm predictions
	 	\param nGroundTruthDetections Add ground truth detections.
	 */
	PrecisionRecall(const std::vector<float>& gt, const std::vector<float>& preds, int nGroundTruthDetections = -1);

	//! Returns area under the curve
	double getAveragePrecision() const { return _averagePrecision; }

	//! Find threshold that results in highest F1 measure
	double getBestThreshold() const;

	//! Save the curve points
	/*! 
		Save curve points to a .pr file. See inclded plot_pr.m file for a
		MATLAB script that can plot this data.
	*/
	void save(const char* filename) const;
};

#endif // PRECISION_RECALL_H
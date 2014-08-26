#include "PrecisionRecall.h"

using namespace std;

static void computePrecisionRecallForThreshold(const vector<float> &gt, const vector<float>& preds, 
	                               float threshold, int nGroundTruthDetections, float& precision, float& recall)
{
	// Compute tp, fp, fn, tn
	int truePos = 0, trueNeg = 0, falsePos = 0, falseNeg = 0;
	for(int i = 0; i < preds.size(); i++) {
		if(preds[i] > threshold) {
			if(gt[i] > 0) truePos++;
			else falsePos++;
		} else if(preds[i] <= threshold) {
			if(gt[i] < 0) trueNeg++;
			else falseNeg++;			
		}
		cout << "Preds["<<i<<"]: " << preds[i] << " threshold: " << threshold << " gt["<<i<<"]: "<< gt[i] <<" truePos: " << truePos << " falsePos: " << falsePos << endl;
	}

	if(truePos + falsePos == 0) precision = 1.0;
	else precision = float(truePos) / (truePos + falsePos);

	int nGt = (nGroundTruthDetections >= 0)? nGroundTruthDetections:(truePos + falseNeg);

	if(truePos + falseNeg == 0) recall = 1.0;
	else recall = float(truePos) / nGt;

	cout << "threshold: "<< threshold <<" truePos: " << truePos << " falsePos: " << falsePos << endl;
}

bool sortByRecall(const PrecisionRecallPoint& a, const PrecisionRecallPoint& b)
{
	return a.recall < b.recall;
}

PrecisionRecall::PrecisionRecall(const std::vector<float> &gt, const std::vector<float>& preds, int nGroundTruthDetections)
{
	vector<float> truePos;
	vector<float> falsePos;
	vector<float> falseNeg;
	vector<float> trueNeg;

	for(int i = 0; i < gt.size(); i++)
	{
		if(gt[i] == preds[i])
		{
			truePos.push_back(1);
			falsePos.push_back(0);
		}
		else
		{
			truePos.push_back(0);
			falsePos.push_back(1);
		}
	}

	for(int i = 1; i < truePos.size(); i++)
	{
		truePos[i] += truePos[i-1];
		falsePos[i] += falsePos[i-1];
	}

	for(int i = 0; i < truePos.size(); i++)
	{
		PrecisionRecallPoint pr;
		pr.recall = truePos[i]/truePos.size();
		pr.precision = truePos[i]/(truePos[i]+falsePos[i]);
		_data.push_back(pr);
	}

	std::sort(_data.begin(), _data.end(), sortByRecall);

	// Remove jags in precision recall curve
	// float maxPrecision = -1;
	// for(std::vector<PrecisionRecallPoint>::reverse_iterator pr = _data.rbegin(); pr != _data.rend(); pr++) {
	// 	pr->precision = std::max(maxPrecision, pr->precision);
	// 	maxPrecision = std::max(pr->precision, maxPrecision);
	// }

	// Compute average precision as area under the curve
	_averagePrecision = 0.0;
	for(std::vector<PrecisionRecallPoint>::iterator pr = _data.begin() + 1, prPrev = _data.begin(); pr != _data.end(); pr++, prPrev++) {
		float xdiff = pr->recall - prPrev->recall;
		float ydiff = pr->precision - prPrev->precision;

		_averagePrecision += xdiff * prPrev->precision + xdiff * ydiff / 2.0;
	}
}

void PrecisionRecall::save(const char* filename) const
{
	std::ofstream f(filename);

	if(f.bad()) throw std::runtime_error("Could not open file " + (std::string)filename + " for writing");

	f << "# precision recall threshold\n";
	for(std::vector<PrecisionRecallPoint>::const_iterator pr = _data.begin(); pr != _data.end(); pr++) {
		f << pr->precision << " " << pr->recall << "\n";
	}
}

double  PrecisionRecall::getBestThreshold() const
{
	double bestFMeasure = -1, bestThreshold = -1;
	for(std::vector<PrecisionRecallPoint>::const_iterator pr = _data.begin(); pr != _data.end(); pr++) {
		double fMeasure = (pr->precision * pr->recall) / (pr->precision + pr->recall);

		if(fMeasure > bestFMeasure) {
			bestFMeasure = fMeasure;
			bestThreshold = pr->threshold;
		}
	}	

	return bestThreshold;
}

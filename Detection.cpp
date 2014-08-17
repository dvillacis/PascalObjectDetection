#include "Detection.h"

using namespace cv;
using namespace std;

Detection::Detection():
    rect(Rect(0,0,0,0)), response(0)
{}

Detection::Detection(Rect rect, double response):
    rect(rect), response(response)
{}

double Detection::area() const
{
    //return width * height;
    return rect.area();
}

double Detection::relativeOverlap(const Detection &other) const
{
    /******** BEGIN TODO ********/
    // Compute the relative overlap between two detections. This
    // is defined as the ratio
    //
    //                    AreaInter(r1, r2)
    //                   ------------------
    //                    AreaUnion(r1, r2)
    //
    // where AreaInter is the area of the intersection of the two
    // rectangles r1 and r2, and AreaUnion is the area of the union
    // of the two rectangles.

    double relOver = 0.0;

printf("TODO: %s:%d\n", __FILE__, __LINE__); 

    /******** END TODO ********/
    return relOver;
}

void Detection::draw(Mat &img) const
{
    rectangle(img,rect,Scalar(255,0,0),2);
    // static const Color red(255, 50, 50);
    // // Bounding box
    // drawHorizLine(img, x - width / 2.0, x + width / 2.0, y - height / 2.0, red);
    // drawHorizLine(img, x - width / 2.0, x + width / 2.0, y + height / 2.0, red);

    // drawVertLine(img, x - width / 2.0, y - height / 2.0, y + height / 2.0, red);
    // drawVertLine(img, x + width / 2.0, y - height / 2.0, y + height / 2.0, red);

    // // Central point
    // const int crossSize = 3;
    // drawHorizLine(img, x - crossSize, x + crossSize, y, red);
    // drawVertLine(img, x, y - crossSize, y + crossSize, red);
}

void drawDetections(Mat &img, const vector<Detection> &dets)
{
    for(vector<Detection>::const_iterator det = dets.begin(), detEnd = dets.end(); det != detEnd; det++) {
        det->draw(img);
    }
}

ostream & operator << (ostream &s, const Detection &d)
{
    s << d.rect << " " << d.response;
    return s;
}

bool sortByResponse(const pair<int, float> &a, const pair<int, float> &b)
{
    return a.second > b.second;
}

void computeLabels(const vector<Detection> &gt, const vector<Detection> &found,
              vector<float> &label, vector<float> &response)
{
    const double overlapThresh = 0.5;

    vector<float> labels;
    vector<pair<int, float> > idxResp(found.size());
    for (int i = 0; i < found.size(); i++) {
        idxResp[i] = pair<int, float>(i, found[i].response);
    }

    sort(idxResp.begin(), idxResp.end(), sortByResponse);

    label = vector<float>(found.size(), -1);
    response = vector<float>(found.size(), 0);

    vector<bool> taken(gt.size(), false);
    for (int i = 0; i < idxResp.size(); i++) {

        int idx = idxResp[i].first;

        const Detection &det = found[idx];

        assert(det.response == idxResp[i].second);

        response[idx] = det.response;

        float bestIdx = -1;
        float bestOverlap = -1;

        for (int j = 0; j < gt.size(); j++) {
            if(taken[j]) continue;

            float overlap = gt[j].relativeOverlap(det);
            if(overlap < overlapThresh) continue;

            if((bestIdx < 0) || (overlap > bestOverlap)) {
                bestOverlap = overlap;
                bestIdx = j;
            }
        }

        if(bestIdx >= 0) {
            label[idx] = 1;
            taken[bestIdx] = true;
        }
    }
}

void computeLabels(const vector<vector<Detection> > &gt, const vector<vector<Detection> > &found,
              vector<float> &label, vector<float> &response, int& nDets)
{
    
    assert(gt.size() == found.size());

    response.resize(0);
    label.resize(0);
    nDets = 0;

    for(int i = 0; i < gt.size(); i++) {
        nDets += gt[i].size();

        vector<float> resp, lab;
        computeLabels(gt[i], found[i], lab, resp);

        int nCorrect = 0;
        for(int j = 0; j < lab.size(); j++) {
            if(lab[j] > 0) nCorrect++;
        }

        LOG(INFO) << nCorrect << "/" << lab.size() << " detections matched";

        response.insert(response.end(), resp.begin(), resp.end());
        label.insert(label.end(), lab.begin(), lab.end());
    }
}




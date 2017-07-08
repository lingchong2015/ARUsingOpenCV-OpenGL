#ifndef LCFEATUREDETECTOR_HPP
#define LCFEATUREDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

class LCFeatureDetector {

public:
	
    LCFeatureDetector(string filename, int minHessian);

    void detectUsingSURF();

    void showTrainSample();

	void matchUsingBFMWithSURF(string testFilename);

	void LCFeatureDetector::matchUsingFLANNWithSURF(string queryFileNmae);

private:
    SurfFeatureDetector mSurfFeatureDectector;

    Mat mTrainImg;

    vector<KeyPoint> mKeyPointsTrain;

	vector<KeyPoint> getQueryKeyPoints(string queryFilename, Mat& queryImg);

	Mat getDescriptorUsingSURF(const Mat& img, vector<KeyPoint> keyPoints);

	void drawMatchsOnWindow(const Mat& queryImg, const vector<KeyPoint>& keyPointsQuery, const vector<DMatch>& matches, string windowName);
};

#endif // LCFEATUREDETECTOR_HPP

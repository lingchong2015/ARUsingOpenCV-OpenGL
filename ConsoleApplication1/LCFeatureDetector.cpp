#include "stdafx.h"
#include <opencv2/legacy/legacy.hpp>
#include "LCFeatureDetector.hpp"

LCFeatureDetector::LCFeatureDetector(string filename, int minHessian) {
	assert(filename != "");

	mTrainImg = imread(filename);
	assert(mTrainImg.data != NULL);

	mSurfFeatureDectector = SurfFeatureDetector(minHessian);
}

void LCFeatureDetector::detectUsingSURF() {
	mSurfFeatureDectector.detect(mTrainImg, mKeyPointsTrain);
}

void LCFeatureDetector::showTrainSample() {
	Mat matWithKeyPoints;
	drawKeypoints(mTrainImg, mKeyPointsTrain, matWithKeyPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SURF Detection", matWithKeyPoints);
}

void LCFeatureDetector::matchUsingBFMWithSURF(string queryFilename) {
	assert(mKeyPointsTrain.size() > 0);

	Mat queryImg;
	vector<KeyPoint> keyPointsQuery = getQueryKeyPoints(queryFilename, queryImg);

	Mat descriptorsTrain = getDescriptorUsingSURF(mTrainImg, mKeyPointsTrain);
	Mat descriptorsQuery = getDescriptorUsingSURF(queryImg, keyPointsQuery);

	BruteForceMatcher<L2<float>> bruteForceMatcher;
	vector<DMatch> dmatches;
	bruteForceMatcher.match(descriptorsQuery, descriptorsTrain, dmatches);

	drawMatchsOnWindow(queryImg, keyPointsQuery, dmatches, "SURF & BruteForceMatcher窗口");
}

void LCFeatureDetector::matchUsingFLANNWithSURF(string queryFilename) {
	assert(mKeyPointsTrain.size() > 0);

	Mat queryImg;
	vector<KeyPoint> keyPointsQuery = getQueryKeyPoints(queryFilename, queryImg);

	Mat descriptorsTrain = getDescriptorUsingSURF(mTrainImg, mKeyPointsTrain);
	Mat descriptorsQuery = getDescriptorUsingSURF(queryImg, keyPointsQuery);

	FlannBasedMatcher flannBasedMatcher;
	vector<DMatch> dmatches;
	flannBasedMatcher.match(descriptorsQuery, descriptorsTrain, dmatches);

	double maxDist = 0;
	double minDist = 100;
	for (int i = 0; i < descriptorsQuery.rows; ++i)
	{
		double dist = dmatches[i].distance;
		if (dist < minDist) {
			minDist = dist;
		}
		if (dist > maxDist) {
			maxDist = dist;
		}
	}

	cout << "最大距离：" << maxDist << endl;
	cout << "最小距离" << minDist << endl;

	vector<DMatch> goodMatches;
	for (int i = 0; i < descriptorsQuery.rows; ++i)
	{
		if (dmatches[i].distance < 2 * minDist) {
			goodMatches.push_back(dmatches[i]);
		}
	}

	drawMatchsOnWindow(queryImg, keyPointsQuery, dmatches, "SURF & FLANNBasedMatcher窗口");
}

vector<KeyPoint> LCFeatureDetector::getQueryKeyPoints(string queryFilename, Mat& queryImg) {
	queryImg = imread(queryFilename);
	assert(queryImg.data != NULL);

	vector<KeyPoint> keyPointsQuery;
	mSurfFeatureDectector.detect(queryImg, keyPointsQuery);

	return keyPointsQuery;
}

Mat LCFeatureDetector::getDescriptorUsingSURF(const Mat& img, vector<KeyPoint> keyPoints) {
	SurfDescriptorExtractor surfDescriptorExtractor;
	Mat descriptor;
	surfDescriptorExtractor.compute(img, keyPoints, descriptor);
	return descriptor;
}

void LCFeatureDetector::drawMatchsOnWindow(const Mat& queryImg, const vector<KeyPoint>& keyPointsQuery, const vector<DMatch>& matches, string windowName) {
	Mat showImg;
	drawMatches(queryImg, keyPointsQuery, mTrainImg, mKeyPointsTrain, matches, showImg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	namedWindow(windowName, CV_WINDOW_NORMAL);
	imshow(windowName, showImg);
}
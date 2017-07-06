#include "stdafx.h"
#include <opencv2/legacy/legacy.hpp>
#include "LCFeatureDetector.hpp"

LCFeatureDetector::LCFeatureDetector(string filename, int minHessian) {
    assert(filename != "");

    mImg = imread(filename);
    assert(mImg.data != NULL);

    mSurfFeatureDectector = SurfFeatureDetector(minHessian);
}

void LCFeatureDetector::detect() {
    mSurfFeatureDectector.detect(mImg, mKeyPoints);
}

void LCFeatureDetector::showSampleDetect() {
    Mat matWithKeyPoints;
    drawKeypoints(mImg, mKeyPoints, matWithKeyPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SURF Detection", matWithKeyPoints);
}

void LCFeatureDetector::match(string testFilename) {
	assert(mKeyPoints.size() > 0);

	Mat testImg = imread(testFilename);
	assert(testImg.data != NULL);
	
	vector<KeyPoint> keyPointsTest;
	mSurfFeatureDectector.detect(testImg, keyPointsTest);

	SurfDescriptorExtractor surfDescriptorExtractor;
	Mat descriptorsSample;
	Mat descriptorsTest;
	surfDescriptorExtractor.compute(mImg, mKeyPoints, descriptorsSample);
	surfDescriptorExtractor.compute(testImg, keyPointsTest, descriptorsTest);

	BruteForceMatcher<L2<float>> bruteForceMatcher;
	vector<DMatch> dmatches;
	bruteForceMatcher.match(descriptorsSample, descriptorsTest, dmatches);

	Mat showImg;
	drawMatches(mImg, mKeyPoints, testImg, keyPointsTest, dmatches, showImg);

	namedWindow("BruteForceMather∆•≈‰", CV_WINDOW_NORMAL);
	imshow("BruteForceMather∆•≈‰", showImg);
}

#include "stdafx.h"
#include "LCFeatureDetector.hpp"

LCFeatureDetector::LCFeatureDetector(std::string filename, int minHessian) {
    assert(filename != "");

    mImg = imread(filename);
    assert(mImg.data != NULL);

    mSurfFeatureDectector = SurfFeatureDetector(minHessian);
}

void LCFeatureDetector::detect() {
    mSurfFeatureDectector.detect(mImg, mKeyPoints);
}

void LCFeatureDetector::show() {
    Mat matWithKeyPoints;
    drawKeypoints(mImg, mKeyPoints, matWithKeyPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SURF Detection", matWithKeyPoints);
}

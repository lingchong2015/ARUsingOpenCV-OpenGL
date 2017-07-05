#ifndef SURFDETECTOR_HPP
#define SURFDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

class LCFeatureDetector {

public:
    LCFeatureDetector(string filename, int minHessian);

    void detect();

    void show();

private:
    SurfFeatureDetector mSurfFeatureDectector;

    Mat mImg;

    vector<KeyPoint> mKeyPoints;
};

#endif // SURFDETECTOR_HPP

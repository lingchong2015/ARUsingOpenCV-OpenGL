// ConsoleApplication1.cpp : 定义控制台应用程序的入口点。
//
/////////////////////////C++ Header File/////////////////////////////////////////
#include "stdafx.h"
#include <iostream>  

//////////////////////////OpenCV Header File/////////////////////////////////////
#include <opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include "CameraCalibration.hpp"
#include "ARPipeline.hpp"
#include "LCFeatureDetector.hpp"

///////////////////////////OpenGL Header File////////////////////////////////////
#include <../../../Microsoft Visual Studio 14.0/VC/include/glut.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

using namespace std;
using namespace cv;

/************************************************************************/
/*					   Markerless AR Begin                              */
/************************************************************************/

string ToString(const float value)
{
	ostringstream oss;
	oss << value;
	return oss.str();
}

/**
* @brief processFrame 对摄像头读入的视频帧进行全方位的检测。
* @param cameraFrame 摄像头输入的视频帧对象。
* @param pipeline AR管道对象。
* @return 如果检测过程能够被停止，则返回true，否则返回false。
*/
bool processFrame(const Mat& cameraFrame, ARPipeline& pipeline) {
	// 深度复制，图像对象中的数据也会被复制，最终的OpenGL图形会覆盖在displayFrame对象上面。
	Mat displayFrame = cameraFrame.clone();

	// Draw information:
	if (pipeline.m_patternDetector.enableHomographyRefinement) {
		putText(displayFrame, "Pose refinement: On   ('h' to switch off)", Point(10, 15), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 200, 0));
	}
	else {
		putText(displayFrame, "Pose refinement: Off  ('h' to switch on)", Point(10, 15), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 200, 0));
	}

	putText(displayFrame, "RANSAC threshold: " + ToString(pipeline.m_patternDetector.homographyReprojectionThreshold) +
		"( Use'-'/'+' to adjust)", Point(10, 30), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 200, 0));

	// 显示窗口界面。
	imshow("LCARDemo", displayFrame);

	// Set a new camera frame:
	// drawingCtx.updateBackground(displayFrame);

	// Find a pattern and update it's detection status:
	//    drawingCtx.isPatternPresent = pipeline.processFrame(cameraFrame);
	bool isFound = pipeline.processFrame(cameraFrame);
	cout << "Pattern is " << (isFound == true ? "match" : "not match") << endl;
	cout << isFound << endl;
	//    cout << isFound << endl;

	// Update a pattern pose:
	//    drawingCtx.patternPose = pipeline.getPatternLocation();
	pipeline.getPatternLocation();

	// Request redraw of the window:
	//    drawingCtx.updateWindow();

	// Read the keyboard input:
	int keyCode = waitKey(5);

	bool shouldQuit = false;
	if (keyCode == '+' || keyCode == '=')
	{
		pipeline.m_patternDetector.homographyReprojectionThreshold += 0.2f;
		pipeline.m_patternDetector.homographyReprojectionThreshold = min(10.0f, pipeline.m_patternDetector.homographyReprojectionThreshold);
	}
	else if (keyCode == '-')
	{
		pipeline.m_patternDetector.homographyReprojectionThreshold -= 0.2f;
		pipeline.m_patternDetector.homographyReprojectionThreshold = max(0.0f, pipeline.m_patternDetector.homographyReprojectionThreshold);
	}
	else if (keyCode == 'h')
	{
		pipeline.m_patternDetector.enableHomographyRefinement = !pipeline.m_patternDetector.enableHomographyRefinement;
	}
	else if (keyCode == 27 || keyCode == 'q')
	{
		shouldQuit = true;
	}

	return shouldQuit;
}

/**
* @brief processVideo 用于处理从摄像头从读入的视频流，并可实时调整单应性提纯与二次投射阈值。
* @param patternImage 模版图像对象。
* @param calibration 摄像机校准对象。
* @param capture 视频捕捉对象。
*/
void processVideo(const Mat& patternImage, CameraCalibration& calibration) {
	VideoCapture capture(0);

	// 先抓取一帧获得视频的尺寸。
	Mat currentFrame;
	capture >> currentFrame;

	if (currentFrame.empty())
	{
		cout << "无法打开摄像头设备。" << endl;
		return;
	}

	Size frameSize(currentFrame.cols, currentFrame.rows);

	// 检测模版图像的特征点，计算特征描述子，训练一个特征描述子匹配器。
	ARPipeline pipeline(patternImage, calibration);
	// OpenGL初始化。
	// ARDrawingContext drawingCtx("Markerless AR", frameSize, calibration);

	bool shouldQuit = false;
	do
	{
		capture >> currentFrame;
		if (currentFrame.empty())
		{
			shouldQuit = true;
			continue;
		}

		shouldQuit = processFrame(currentFrame, pipeline);
	} while (!shouldQuit);
}

void processSingleImage(const Mat& patternImage, CameraCalibration& calibration, const Mat& image)
{
	Size frameSize(image.cols, image.rows);
	ARPipeline pipeline(patternImage, calibration);
	//    ARDrawingContext drawingCtx("Markerless AR", frameSize, calibration);

	bool shouldQuit = false;
	do
	{
		shouldQuit = processFrame(image, pipeline);
	} while (!shouldQuit);
}

void markerlessAR() {
	// Change this calibration to yours:
	CameraCalibration calibration(526.58037684199849f, 524.65577209994706f, 318.41744018680112f, 202.96659047014398f);

	// Try to read the pattern:
	Mat patternImage = imread("1.jpg");
	if (patternImage.empty())
	{
		cout << "模式图像读取失败。" << endl;
		return;
	}

	/*Mat testImage = imread("1.jpg");
	if (!testImage.empty())
	{
		processSingleImage(patternImage, calibration, testImage);
	}*/

	processVideo(patternImage, calibration);
}

/************************************************************************/
/*					     Markerless AR End                              */
/************************************************************************/

/************************************************************************/
/*                        TrackBar Test Begin                           */
/************************************************************************/
Mat gSrcImg1;
Mat gSrcImg2;

const string TRACK_BAR_WINDOW_NAME = "线性混合窗口";
const string TRACK_BAR_NAME = "";

void onTrackBarChanged(int i, void* data) {
	double alphaRatio = (double)i / 100;
	double betaRatio = 1 - alphaRatio;

	Mat* showImg = (Mat*)data;

	addWeighted(gSrcImg1, alphaRatio, gSrcImg2, betaRatio, 0, *showImg);
	imshow(TRACK_BAR_WINDOW_NAME, *showImg);
}

void TrackBarTest() {
	gSrcImg1 = imread("1.jpg");
	gSrcImg2 = imread("2.jpg");

	assert(gSrcImg1.data != NULL);
	assert(gSrcImg2.data != NULL);

	int alphaValue = 70;
	namedWindow(TRACK_BAR_WINDOW_NAME, CV_WINDOW_AUTOSIZE);

	Mat showImg = Mat();
	createTrackbar(TRACK_BAR_NAME, TRACK_BAR_WINDOW_NAME, &alphaValue, 100, onTrackBarChanged, (void*)&showImg);

	onTrackBarChanged(70, (void*)&showImg);

	waitKey(0);
}

/************************************************************************/
/*                        TrackBar Test End                             */
/************************************************************************/

/************************************************************************/
/*					 Contrast & Bright Test Begin                       */
/************************************************************************/

const string CONTRAST_AND_BRIGHT_WINDOW_TITLE = "屏幕对比度与亮度测试";
const string CONTRAST_TRACK_BAR_TITLE = "对比度";
const string BRIGHT_TRACK_BAR_TITLE = "亮度";

int gnContrast = 80;
int gnBright = 80;
Mat gImgSrc;
Mat gImgShow;

void onContrastAndBrightTrackBarChanged(int value, void* userData) {
	for (int row = 0; row < gImgSrc.rows; ++row)
	{
		for (int col = 0; col < gImgSrc.cols; ++col)
		{
			for (int elem = 0; elem < 3; ++elem)
			{
				gImgShow.at<Vec3b>(row, col)[elem] = saturate_cast<uchar>(gnContrast * 0.01 * (gImgSrc.at<Vec3b>(row, col)[elem]) + gnBright);
			}
		}
	}

	imshow(CONTRAST_AND_BRIGHT_WINDOW_TITLE, gImgShow);
}

void contrastAndBrightTest() {
	gImgSrc = imread("1.jpg");
	gImgShow = Mat::zeros(gImgSrc.size(), gImgSrc.type());

	namedWindow(CONTRAST_AND_BRIGHT_WINDOW_TITLE, CV_WINDOW_AUTOSIZE);

	createTrackbar(CONTRAST_TRACK_BAR_TITLE, CONTRAST_AND_BRIGHT_WINDOW_TITLE, &gnContrast, 300, onContrastAndBrightTrackBarChanged, 0);
	createTrackbar(BRIGHT_TRACK_BAR_TITLE, CONTRAST_AND_BRIGHT_WINDOW_TITLE, &gnBright, 200, onContrastAndBrightTrackBarChanged, 0);

	onContrastAndBrightTrackBarChanged(0, 0);

	waitKey(0);
}

/************************************************************************/
/*					 Contrast & Bright Test End                         */
/************************************************************************/

/************************************************************************/
/*					Threshold Test Begin                                */
/************************************************************************/

#define THRESHOLD_WINDOW_TITLE "阈值测试"
#define THRESHOLD_MODE_TRACK_BAR_TITLE "阈值模式"
#define THRESHOLD_VALUE_TRACK_BAR_TITLE "阈值大小"

int gnThresholdType = 3;
int gnThresholdValue = 100;

Mat gImgGray;
Mat gImgThreshold;
void onThresholdTrackBarChanged(int value, void* pUserData) {
	threshold(gImgGray, gImgThreshold, gnThresholdValue, 255, gnThresholdType);
	imshow(THRESHOLD_WINDOW_TITLE, gImgThreshold);
}

void thresholdTest() {
	Mat imgSrc = imread("1.jpg");
	cvtColor(imgSrc, gImgGray, CV_RGB2GRAY);
	gImgThreshold = Mat::zeros(imgSrc.size(), imgSrc.type());

	namedWindow(THRESHOLD_WINDOW_TITLE, CV_WINDOW_NORMAL);

	createTrackbar(THRESHOLD_MODE_TRACK_BAR_TITLE, THRESHOLD_WINDOW_TITLE, &gnThresholdType, 4, onThresholdTrackBarChanged, 0);
	createTrackbar(THRESHOLD_VALUE_TRACK_BAR_TITLE, THRESHOLD_WINDOW_TITLE, &gnThresholdValue, 255, onThresholdTrackBarChanged, 0);

	onThresholdTrackBarChanged(0, 0);

	waitKey(0);
}

/************************************************************************/
/*					 Threshold Test End                                 */
/************************************************************************/

/************************************************************************/
/*                   Canny Edge Detection Begin                        */
/************************************************************************/

#define CANNY_DETECTION_WINDOW_TITLE "Canny边缘检测"

void simpleCannyDetection() {
	Mat imgSrc = imread("1.jpg");
	Canny(imgSrc, imgSrc, 75, 150, 3);
	imshow(CANNY_DETECTION_WINDOW_TITLE, imgSrc);
	waitKey(0);
}

void finnerCannyDetection() {
	Mat imgSrc = imread("1.jpg");
	Mat imgShow;
	imgShow = Mat::zeros(imgSrc.size(), imgSrc.type());
	Mat imgGray;
	cvtColor(imgSrc, imgGray, CV_RGB2GRAY);// Source => Gray.
	Mat imgEdge;
	blur(imgGray, imgEdge, Size(3, 3));// Noise reduction.
	Canny(imgEdge, imgEdge, 90, 150, 3);// Canny detection.
	imgSrc.copyTo(imgShow, imgEdge);

	imshow(CANNY_DETECTION_WINDOW_TITLE, imgShow);
	waitKey(0);
}

/************************************************************************/
/*                    Canny Edge Detection End                          */
/************************************************************************/

/************************************************************************/
/*						OpenGL Test Begin                               */
/************************************************************************/

static int year = 0, spin = 0, day = 0;
static GLint fogMode;
const int n = 100;
const GLfloat R = 1.0f;
const GLfloat Pi = 3.1415926536f;

void DrawCircle() {

	int  i;
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_LINE_LOOP);

	for (i = 0; i < n; ++i)
	{
		glColor3f(1.0, 0.0, 0.0);
		glVertex2f(R*cos(2 * Pi / n*i), R*sin(2 * Pi / n*i));
	}

	glEnd();
	glFlush();
}

void init(void) {
	GLfloat position[] = { 0.5, 0.5, 3.0, 0.0 };
	glEnable(GL_DEPTH_TEST);                          //防止遮挡
	glLightfv(GL_LIGHT0, GL_POSITION, position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	{
		GLfloat mat[3] = { 0.1745, 0.01175, 0.01175 };
		glMaterialfv(GL_FRONT, GL_AMBIENT, mat);
		mat[0] = 0.61424; mat[1] = 0.04136; mat[2] = 0.04136;
		glMaterialfv(GL_FRONT, GL_DIFFUSE, mat);
		mat[0] = 0.727811; mat[1] = 0.626959; mat[2] = 0.626959;
		glMaterialfv(GL_FRONT, GL_SPECULAR, mat);
		glMaterialf(GL_FRONT, GL_SHININESS, 0.6*128.0);
	}

	glEnable(GL_FOG);

	{
		GLfloat fogColor[4] = { 0.5, 0.5, 0.5, 1.0 };
		fogMode = GL_EXP;
		glFogi(GL_FOG_MODE, fogMode);
		glFogfv(GL_FOG_COLOR, fogColor);
		glFogf(GL_FOG_DENSITY, 0.35);
		glHint(GL_FOG_HINT, GL_DONT_CARE);
		glFogf(GL_FOG_START, 1.0);
		glFogf(GL_FOG_END, 5.0);
	}

	glClearColor(0.5, 0.9, 0.9, 1.0);  /* fog color */

}

void display(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColor3f(0.0, 1.0, 1.0);
	glPushMatrix(); //记住自己的位置
	glutSolidSphere(1.0, 20, 16);   /* 画太阳半径、 20经度、16纬度*/
	glRotatef(spin, 0.0, 1.0, 0.0);  //自转，绕着一个向量以给定角度旋转（正的为逆时针）
	glTranslatef(2.0, 1.0, 0.0);
	glRotatef(spin, 1.0, 0.0, 0.0); //公转
	glRectf(0.1, 0.1, 0.5, 0.5);
	glColor3f(0.0, 0.0, 1.0);
	glutWireSphere(0.2, 8, 8);    /* 画第一颗小行星 */
	glColor3f(1.0, 0.0, 0.0);
	glTranslatef(2.0, 1.0, 0.0);
	glRotatef(2 * spin, 0.0, 1.0, 0.0);
	glutSolidSphere(0.5, 16, 8);
	glPopMatrix();//回到原来的位置
	glutSwapBuffers();
}

void spinDisplay(void) {
	spin = spin + 2;
	if (spin > 360)
		spin = spin - 360;
	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		if (state == GLUT_DOWN)
			glutIdleFunc(spinDisplay);
		break;

	case GLUT_MIDDLE_BUTTON:
		if (state == GLUT_DOWN)
			glutIdleFunc(NULL);
		break;

	default:
		break;
	}

}

void reshape(int w, int h) {
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 0.5, 20.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

void keyboard(unsigned char key, int x, int y) {
	switch (key) {
	case 'd':
		day = (day + 10) % 360;
		glutPostRedisplay();
		break;
	case 'D':
		day = (day - 10) % 360;
		glutPostRedisplay();
		break;
	case 'y':
		year = (year + 5) % 360;
		glutPostRedisplay();
		break;
	case 'Y':
		year = (year - 5) % 360;
		glutPostRedisplay();
		break;
	case 27:
		exit(0);
		break;
	default:
		break;
	}
}

/************************************************************************/
/*						OpenGL Test End                                 */
/************************************************************************/

/************************************************************************/
/*						Filter Test Begin                               */
/************************************************************************/

void filter() {
	Mat src = imread("1.jpg");

	string s1 = "方框滤波测试";
	namedWindow(s1, CV_WINDOW_AUTOSIZE);

	Mat out;
	boxFilter(src, out, -1, Size(11, 11));
	imshow(s1, out);

	string s2 = "均值滤波测试";
	namedWindow(s2, CV_WINDOW_AUTOSIZE);

	blur(src, out, Size(11, 11));
	imshow(s2, out);

	string s3 = "高斯滤波测试";
	namedWindow(s3, CV_WINDOW_AUTOSIZE);

	GaussianBlur(src, out, Size(11, 11), 0, 0);
	imshow(s3, out);

	string s4 = "中值滤波测试";
	namedWindow(s4, CV_WINDOW_AUTOSIZE);

	medianBlur(src, out, 11);
	imshow(s4, out);

	string s5 = "双边滤波测试";
	namedWindow(s5, CV_WINDOW_AUTOSIZE);

	bilateralFilter(src, out, 10, 10 * 2, 10 / 2);
	imshow(s5, out);

	waitKey(0);
}

/************************************************************************/
/*						 Filter Test End                                */
/************************************************************************/

/************************************************************************/
/*				 Feature Detection & Match Begin                        */
/************************************************************************/

void suffDetect() {
	LCFeatureDetector lcFeatureDetector = LCFeatureDetector("1.jpg", 400);
	lcFeatureDetector.detect();
	lcFeatureDetector.show();

	waitKey(0);
}

/************************************************************************/
/*				  Feature Detection & Match End                         */
/************************************************************************/

/************************************************************************/
/*						main Function Begin                             */
/************************************************************************/

int main(int argc, char** argv)
{
	markerlessAR();

	//TrackBarTest();

	//contrastAndBrightTest();

	//thresholdTest();

	//simpleCannyDetection();

	//finnerCannyDetection();

	/*
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(400, 400);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Hello World, OpenGL!");
	init();
	//glutDisplayFunc(DrawCircle);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	//glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMainLoop();
	*/

	//filter();

	//suffDetect();

	return 0;
}

/************************************************************************/
/*						  main Function End                             */
/************************************************************************/
/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#define _copysign copysign


#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <vector>
#include <iostream>


#include "HoG.h"

// structure for image patch
struct PatchFeature {
	PatchFeature() {}

	CvRect roi;
	std::vector<CvPoint> center;
	std::vector<CvMat*> vPatch;
	bool fg;
	double err;
	double cmean;

	void print() const {
		std::cout << roi.x << " " << roi.y << " " << roi.width << " " << roi.height;
		for(unsigned int i=0; i<center.size(); ++i) std::cout << " " << center[i].x << " " << center[i].y; std::cout << std::endl;
	}
	void show(int delay) const;
};

struct LessThanFeature {
	bool operator() (const PatchFeature& p1, const PatchFeature& p2) const {
		try{
			if (p1.fg && p2.fg) return (p1.err > p2.err);
			else if (!p1.fg && !p2.fg) return (p1.cmean > p2.cmean);
			else throw 2;
		}
		catch (int e){
			std::cout << "You're mixing forground and background patches in Cascade!" << std::endl;
		}
	}
};

// structure for image patch for negative mining
struct PatchHardMining {
	PatchHardMining() {}
	std::string patchpath;
	double cmean;
};

struct LessThanPatchHardMining {
	bool operator() (const PatchHardMining& p1, const PatchHardMining& p2) const {
		return (p1.cmean > p2.cmean);
	}
};

static HoG hog; 

class CRPatch {
public:
	CRPatch(CvRNG* pRNG, int w, int h, int num_l) : cvRNG(pRNG), width(w), height(h) { vLPatches.resize(num_l);}

	// Extract patches from image
	void extractPatches_orig(IplImage *img, unsigned int n, int label, CvRect* box = 0, std::vector<CvPoint>* vCenter = 0);
	void extractPatches(IplImage *img, const char* fullpath, unsigned int n, int label, CvRect* box = 0, std::vector<CvPoint>* vCenter = 0);
	void extractPatchesMul(IplImage *img, const char* fullpath, unsigned int n, int label, std::vector<CvRect>& box, int startpos,int endpos);
	// Extract features from image
	static void extractFeatureChannels(IplImage *img, std::vector<IplImage*>& vImg);
	// Extract features from image ( Only HOG like features - and adding my PCAs and PCAm)
	static void extractFeatureChannelsPartial(IplImage *img, std::vector<IplImage*>& vImg, const char* fullpath);
	// Extract features from image ( All Houghforest - and adding my PCAs and PCAm)
	static void CRPatch::extractFeatureChannelsExtra(IplImage *img, std::vector<IplImage*>& vImg, const char* fullpath);
	static void CRPatch::extractPCAChannels(IplImage *img, std::vector<IplImage*>& vImg, const char* fullpath);
	static void CRPatch::extractPCAChannelsPlusEst(IplImage *img, std::vector<IplImage*>& vImg, const char* fullpath);
	// min/max filter
	static void maxfilt(uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width);
	static void maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width);
	static void minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
	static void maxminfilt(uchar* data, uchar* maxvalues, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width);
	static void maxfilt(IplImage *src, unsigned int width);
	static void maxfilt(IplImage *src, IplImage *dst, unsigned int width);
	static void minfilt(IplImage *src, unsigned int width);
	static void minfilt(IplImage *src, IplImage *dst, unsigned int width);
	//Helper functions
	static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat1i &X, cv::Mat1i &Y);
	static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv, cv::Mat1i &X, cv::Mat1i &Y);
	cv::Mat static fixMat2GMM(cv::Mat& nmsmat, cv::Mat& img, int sigma = 15);

	std::vector<std::vector<PatchFeature> > vLPatches;
private:
	CvRNG *cvRNG;
	int width;
	int height;
};


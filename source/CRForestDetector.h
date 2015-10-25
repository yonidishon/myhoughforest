/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRForest.h"
#include <ctime>
#include <queue>
#include <opencv2/core/core.hpp>


class CRForestDetector {
public:
	// Constructor
	CRForestDetector(const CRForest* pRF, int w, int h) : crForest(pRF), width(w), height(h)  {}

	// detect multi scale
	void detectPyramid(IplImage *img, std::vector<std::vector<IplImage*> >& imgDetect, std::vector<float>& ratios,const char* imfile);

	void detectPyramidcascade(PatchFeature& p, std::priority_queue<PatchFeature, std::vector<PatchFeature>,LessThanFeature>& pos_bad_examples,
		std::priority_queue<PatchFeature, std::vector<PatchFeature>, LessThanFeature>& neg_bad_examples,
		int k, std::vector<float>& ratios);
	
	float detectPyramidhard(IplImage *img, std::vector<float>& scales, std::vector<float>& ratios,
		const char *imfile, std::priority_queue<PatchHardMining, std::vector<PatchHardMining>, LessThanPatchHardMining>& neg_examples,
		const char *posfile, CvRect vBBox, CvPoint vCenter, int max_neg_samples);

	// Get/Set functions
	unsigned int GetNumCenter() const {return crForest->GetNumCenter();}

	//Helper For hard negative mining - check if a feature patch is inside the positive BB
	bool isinposrect(CvRect& BB, CvPoint& l_cor);

private:
	// Detectors
	void detectColor(IplImage *img, std::vector<IplImage*>& imgDetect, std::vector<float>& ratios, const char* imfile);

	float detectColorHardNeg(IplImage* img, std::priority_queue<PatchHardMining, std::vector<PatchHardMining>, LessThanPatchHardMining>& pos_bad_examples,
		std::vector<float>& ratios, const char* imfile, const char* filename, CvRect vBBox, CvPoint vCenter, int max_neg_samples);

	void detectColorcascade(PatchFeature& p, std::priority_queue<PatchFeature, std::vector<PatchFeature>, LessThanFeature>& pos_bad_examples,
		std::priority_queue<PatchFeature, std::vector<PatchFeature>, LessThanFeature>& neg_bad_examples,
		int k, std::vector<float>& ratios);


	
	const CRForest* crForest;
	int width;
	int height;
};



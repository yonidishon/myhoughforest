/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRForest.h"
#include <ctime>
#include <queue>

class CRForestDetector {
public:
	// Constructor
	CRForestDetector(const CRForest* pRF, int w, int h) : crForest(pRF), width(w), height(h)  {}

	// detect multi scale
	void detectPyramid(IplImage *img, std::vector<std::vector<IplImage*> >& imgDetect, std::vector<float>& ratios,const char* imfile);
	void CRForestDetector::detectPyramidcascade(PatchFeature& p, std::priority_queue<PatchFeature, std::vector<PatchFeature>,
		LessThanFeature>& pos_bad_examples, std::priority_queue<PatchFeature, std::vector<PatchFeature>, LessThanFeature>& neg_bad_examples, int k, std::vector<float>& ratios);

	// Get/Set functions
	unsigned int GetNumCenter() const {return crForest->GetNumCenter();}

private:
	void detectColor(IplImage *img, std::vector<IplImage*>& imgDetect, std::vector<float>& ratios, const char* imfile);
	void detectColorcascade(PatchFeature& p, std::priority_queue<PatchFeature, std::vector<PatchFeature>, LessThanFeature>& pos_bad_examples,
		std::priority_queue<PatchFeature, std::vector<PatchFeature>, LessThanFeature>& neg_bad_examples, int k, std::vector<float>& ratios);

	const CRForest* crForest;
	int width;
	int height;
};



/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRForest.h"
#include <ctime>

class CRForestDetector {
public:
	// Constructor
	CRForestDetector(const CRForest* pRF, int w, int h) : crForest(pRF), width(w), height(h)  {}

	// detect multi scale
	void detectPyramid(IplImage *img, std::vector<std::vector<IplImage*> >& imgDetect, std::vector<float>& ratios,const char* imfile);

	// Get/Set functions
	unsigned int GetNumCenter() const {return crForest->GetNumCenter();}

private:
	void detectColor(IplImage *img, std::vector<IplImage*>& imgDetect, std::vector<float>& ratios, const char* imfile);

	const CRForest* crForest;
	int width;
	int height;
};

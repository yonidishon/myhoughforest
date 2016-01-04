/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRTree.h"

#include <vector>

class CRForest {
public:
	// Constructors
	CRForest(int trees = 0) {
		vTrees.resize(trees);
	}
	~CRForest() {
		for(std::vector<CRTree*>::iterator it = vTrees.begin(); it != vTrees.end(); ++it) 
			delete *it;
		vTrees.clear();
	}

	// Set/Get functions
	void SetTrees(int n) {vTrees.resize(n);}
	int GetSize() const {return vTrees.size();}
	unsigned int GetDepth() const {return vTrees[0]->GetDepth();}
	unsigned int GetNumCenter() const {return vTrees[0]->GetNumCenter();}
	int getPatchWidth(){ return vTrees[0]->getPatchWidth(); }
	int getPatchHeight(){ return vTrees[0]->getPatchHeight(); }
	int getNoChans(){ return vTrees[0]->getNoChannels(); }

	// Regression 
	void regression(std::vector<const LeafNode*>& result, uchar** ptFCh, int stepImg) const;
	std::vector<const LeafNode*> regressionIntegral(const std::vector< cv::Mat >& patch, const cv::Mat& nonZeros, const cv::Rect& roi) const;
	// Training
	void trainForest(int min_s, int max_d, CvRNG* pRNG, const CRPatch& TrData, int samples);

	// IO functions
	void saveForest(const char* filename, unsigned int offset = 0);
	void loadForest(const char* filename, int type = 0);
	bool loadForestPatch(const char* filename);
	void show(int w, int h) const {vTrees[0]->showLeaves(w,h);}

	// Trees
	std::vector<CRTree*> vTrees;
};
//Regression
inline void CRForest::regression(std::vector<const LeafNode*>& result, uchar** ptFCh, int stepImg) const {
	result.resize( vTrees.size() );
	for(int i=0; i<(int)vTrees.size(); ++i) {
		result[i] = vTrees[i]->regression(ptFCh, stepImg);
	}
}
inline std::vector<const LeafNode*> CRForest::regressionIntegral(const std::vector< cv::Mat >& patch, const cv::Mat& nonZeros, const cv::Rect& roi) const {

	std::vector<const LeafNode*> res;
	for (int i = 0; i<(int)vTrees.size(); ++i)
		res.push_back(vTrees[i]->regressionIntegral(patch, nonZeros, roi));
	return res;
}

//Training
inline void CRForest::trainForest(int min_s, int max_d, CvRNG* pRNG, const CRPatch& TrData, int samples) {
	for(int i=0; i < (int)vTrees.size(); ++i) {
		vTrees[i] = new CRTree(min_s, max_d, TrData.vLPatches[1][0].center.size(), pRNG);
		vTrees[i]->growTree(TrData, samples);
	}
}

// IO Functions
inline void CRForest::saveForest(const char* filename, unsigned int offset) {
	char buffer[200];
	for(unsigned int i=0; i<vTrees.size(); ++i) {
		sprintf_s(buffer,"%s%03d.txt",filename,i+offset);
		vTrees[i]->saveTree(buffer);
	}
}

inline void CRForest::loadForest(const char* filename, int type) {
	char buffer[200];
	for(unsigned int i=0; i<vTrees.size(); ++i) {
		sprintf_s(buffer,"%s%03d.txt",filename,i);
		vTrees[i] = new CRTree(buffer);
	}
}
inline bool CRForest::loadForestPatch(const char* filename) {

	char buffer[200];
	bool success = true;
	for (unsigned int i = 0; i<vTrees.size(); ++i) {
		sprintf(buffer, "%s%03d.bin", filename, i);
		vTrees[i] = new CRTree();
		success &= vTrees[i]->loadTree(buffer);
	}
	return success;
}
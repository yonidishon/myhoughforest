/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

//#define sprintf_s sprintf 

#include "CRPatch.h"
#include <iostream>
#include <fstream>

// Auxilary structure
struct IntIndex {
	double val;
	unsigned int index;
	bool operator<(const IntIndex& a) const { return val<a.val; }
};

// Structure for the leafs
struct LeafNode {
	// Constructors
	LeafNode() {}

	// IO functions
	void show(int delay, int width, int height); 
	void print() const {
		std::cout << "Leaf " << vCenter.size() << " "  << pfg << std::endl;
	}

	// Probability of foreground
	float pfg;
	// Vectors from object center to training patches
	std::vector<std::vector<CvPoint> > vCenter;	
};

class CRTree {
public:
	// Constructors
	CRTree(const char* filename);
	CRTree(int min_s, int max_d, int cp, CvRNG* pRNG) : min_samples(min_s), max_depth(max_d), num_leaf(0), num_cp(cp), cvRNG(pRNG) {
		num_nodes = (int)pow(2.0,int(max_depth+1))-1;
		// num_nodes x NUMCOL matrix as vector
		//treetable = [leafindex,x1,y1,x2,y2,ch,th,testkind]
		treetable = new int[num_nodes * NUMCOL];
		for (unsigned int i = 0; i<num_nodes * NUMCOL; ++i) treetable[i] = 0;
		// allocate memory for leafs
		leaf = new LeafNode[(int)pow(2.0,int(max_depth))];
	}
	~CRTree() {delete[] leaf; delete[] treetable;}

	// Set/Get functions
	unsigned int GetDepth() const {return max_depth;}
	unsigned int GetNumCenter() const {return num_cp;}

	// Regression
	const LeafNode* regression(uchar** ptFCh, int stepImg) const;

	// Training
	void growTree(const CRPatch& TrData, int samples);

	// IO functions
	bool saveTree(const char* filename) const;
	void showLeaves(int width, int height) const {
		for(unsigned int l=0; l<num_leaf; ++l)
			leaf[l].show(5000, width, height);
	}

private: 

	// Private functions for training
	void grow(const std::vector<std::vector<const PatchFeature*> >& TrainSet, int node, unsigned int depth, int samples, float pnratio);
	void makeLeaf(const std::vector<std::vector<const PatchFeature*> >& TrainSet, float pnratio, int node);
	bool optimizeTest(std::vector<std::vector<const PatchFeature*> >& SetA, std::vector<std::vector<const PatchFeature*> >& SetB, const std::vector<std::vector<const PatchFeature*> >& TrainSet, int* test, unsigned int iter, unsigned int mode);
	void generateTest(int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c);
	void generateTestAve(int* test, unsigned int max_w, unsigned int max_h, int chan);
	void evaluateTest(std::vector<std::vector<IntIndex> >& valSet, const int* test, const std::vector<std::vector<const PatchFeature*> >& TrainSet);
	void split(std::vector<std::vector<const PatchFeature*> >& SetA, std::vector<std::vector<const PatchFeature*> >& SetB, const std::vector<std::vector<const PatchFeature*> >& TrainSet, const std::vector<std::vector<IntIndex> >& valSet, int t);
	double measureSet(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB, unsigned int mode) {
	  if (mode==0) return InfGain(SetA, SetB); else return -distMean(SetA[1],SetB[1]);
	}
	double distMean(const std::vector<const PatchFeature*>& SetA, const std::vector<const PatchFeature*>& SetB);
	double InfGain(const std::vector<std::vector<const PatchFeature*> >& SetA, const std::vector<std::vector<const PatchFeature*> >& SetB);


	// Data structure

	// tree table
	// 2^(max_depth+1)-1 x NUMCOL matrix as vector
	// column: leafindex x1 y1 x2 y2 channel thres
	// if node is not a leaf, leaf=-1
	int* treetable;
	// if NUMCOL == 7 => column: leafindex x1 y1 x2 y2 channel thres
	// if NUMCOL == 8 => column: leafindex x1 y1 x2 y2 channel thres testflag (support Ave of all patch test)
	// if NUMCOL == 12 => column: leafindex xa1 ya1 xb1 yb2 xa2 ya2 xb2 yb2 channel thres testflag (support sub-patch tests)
	int NUMCOL = 12;

	// stop growing when number of patches is less than min_samples
	unsigned int min_samples;

	// depth of the tree: 0-max_depth
	unsigned int max_depth;

	// number of nodes: 2^(max_depth+1)-1
	unsigned int num_nodes;

	// number of leafs
	unsigned int num_leaf;

	// number of center points per patch
	unsigned int num_cp;

	//leafs as vector
	LeafNode* leaf;

	CvRNG *cvRNG;
};

inline const LeafNode* CRTree::regression(uchar** ptFCh, int stepImg) const {
	// pointer to current node
	const int* pnode = &treetable[0];
	int node = 0;

	// Go through tree until one arrives at a leaf, i.e. pnode[0]>=0)
	while(pnode[0]==-1) {
		// binary test 0 - left, 1 - right
		// Note that x, y are changed since the patches are given as matrix and not as image 
		// p1 - p2 < t -> left is equal to (p1 - p2 >= t) == false
		// subpatches test pnode = [leafindex xa1 ya1 xb1 yb2 xa2 ya2 xb2 yb2 channel thres testflag]
		//						  [ 	0	  1	  2   3   4   5   6   7   8     9      10     11   ]
		// test has no leafindex and no test flag
		// Choosing the channel
		uchar* ptC = ptFCh[pnode[9]];
	
		// test
		bool test;
		switch (pnode[NUMCOL-1]){ // last column has the type of test currently only pixel based and average of patch
		case 1: {//patch mean
			int patchSum = 0;
			int numel = 0;
			for (int y = pnode[2]; y < pnode[4]+1;y++){
				for (int x = pnode[1]; x < pnode[3] + 1;x++){
					patchSum += *(ptC + x + y*stepImg);
					numel++;
				}
			}
			double patchAve = patchSum / numel;
			test = (patchAve) >= pnode[NUMCOL-2];
			break;
		}
		default: { // subpatches test pnode = [leafindex xa1 ya1 xb1 yb2 xa2 ya2 xb2 yb2 channel thres testflag]
				   //						  [ 	0	  1	  2   3   4   5   6   7   8     9      10     11   ]
				   // test has no leafindex and no test flag

			int patch1sum = 0;
			int patch2sum = 0;
			int numelpatch1 = 0;
			int numelpathc2 = 0;

			//subpatch A
			for (int y = pnode[2]; y < pnode[6] + 1; y++){
				for (int x = pnode[1]; x < pnode[5] + 1; x++){
					patch1sum += *(ptC + x + y * stepImg);
					numelpatch1++;
				}
			}
			double Patch1ave = patch1sum / (double)numelpatch1;

			//subpatch B
			for (int y = pnode[4]; y < pnode[8] + 1; y++){
				for (int x = pnode[3]; x < pnode[7] + 1; x++){
					patch2sum += *(ptC + x + y * stepImg);
					numelpathc2++;
				}
			}
			double Patch2ave = patch2sum / (double)numelpathc2;
			//calculate sub-patch difference
			test = (Patch1ave - Patch2ave) >= pnode[NUMCOL-2];
			break;
		}
		}

		// next node: 2*node_id + 1 + test
		// increment node/pointer by node_id + 1 + test
		int incr = node+1+test;
		node += incr;
		pnode += incr*NUMCOL;
	}

	// return leaf
	return &leaf[pnode[0]];
}
// randomally select two pixels in the patch and the channel
inline void CRTree::generateTest(int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c) {
	//return value is int array of zize[9] [a1(x,y),b1(x,y),a2(x,y),b2(x,y),channel]

	// a1,b1 is the top-left of two sub-patches and a2,b2 defines the right-bottom
	/*	int xa1 = roi.x + pnode[1];		int xa2 = xa1 + pnode[5];
	int ya1 = roi.y + pnode[2];		int ya2 = ya1 + pnode[6];
	int xb1 = roi.x + pnode[3];		int xb2 = xb1 + pnode[7];
	int yb1 = roi.y + pnode[4];		int yb2 = yb1 + pnode[8];*/

	test[0] = cvRandInt(cvRNG) % max_w;				//xa1
	test[1] = cvRandInt(cvRNG) % max_h;				//ya1
	test[2] = cvRandInt(cvRNG) % max_w;				//xb1
	test[3] = cvRandInt(cvRNG) % max_h;				//yb2
	test[4] = MAX(cvRandInt(cvRNG) % (max_w - test[0]),1);	//xa2
	test[5] = MAX(cvRandInt(cvRNG) % (max_h - test[1]),1);	//ya2
	test[6] = MAX(cvRandInt(cvRNG) % (max_w - test[2]),1);	//xb2
	test[7] = MAX(cvRandInt(cvRNG) % (max_h - test[3]),1);	//yb2
	test[8] = cvRandInt(cvRNG) % max_c;
	test[NUMCOL - 2] = 0; // because thersh is between (in test[9]) the flag and roi
}
inline void CRTree::generateTestAve(int* test, unsigned int max_w, unsigned int max_h, int chan) {
	//return value is int array of zize[5] [p1(x,y),p2(x,y),channel]
	// subpatches test pnode = [leafindex xa1 ya1 xb1 yb2 xa2 ya2 xb2 yb2 channel thres testflag]
	//						  [ 	0	  1	  2   3   4   5   6   7   8     9      10     11   ]
	// test has no leafindex and no test flag
	test[0] = 0;
	test[1] = 0;
	test[2] = max_w-1;
	test[3] = max_h-1;
	test[4] = 0;
	test[5] = 0;
	test[6] = 0;
	test[7] = 0;
	test[8] = chan;
	test[NUMCOL - 2] = 1; // because thersh is between (in test[9]) the flag and roi
}

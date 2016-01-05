/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRTree.h"
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <vector>
#include <iostream>

using namespace std;

/////////////////////// Constructors /////////////////////////////

// Read tree from file
CRTree::CRTree(const char* filename) {
	cout << "Load Tree " << filename << endl;

	int dummy;

	ifstream in(filename);
	if(in.is_open()) {
		// allocate memory for tree table
		in >> max_depth;
		num_nodes = (int)pow(2.0,int(max_depth+1))-1;
		// num_nodes x NUM_COL matrix as vector ([leafindex,x1,y1,x2,y2,chan,th]-leafindex (if not leaf == -1),x1,y1,x2,y2 (coordinates for test to perform),channel,thres (chosen in random))
		treetable = new int[num_nodes * NUM_COL];
		int* ptT = &treetable[0];
		
		// allocate memory for leafs
		in >> num_leaf;
		leaf = new LeafNode[num_leaf];

		// number of center points per patch 
		in >> num_cp;

		// read tree nodes
		for(unsigned int n=0; n<num_nodes; ++n) {
			in >> dummy; in >> dummy;
			for (unsigned int i = 0; i<NUM_COL; ++i, ++ptT) {
				in >> *ptT;
			}
		}

		// read tree leafs
		LeafNode* ptLN = &leaf[0];
		for(unsigned int l=0; l<num_leaf; ++l, ++ptLN) {
			in >> dummy;
			in >> ptLN->pfg;
			
			// number of positive patches
			in >> dummy;
			ptLN->vCenter.resize(dummy);
			for(int i=0; i<dummy; ++i) {
				ptLN->vCenter[i].resize(num_cp);
				for(unsigned int k=0; k<num_cp; ++k) {
					in >> ptLN->vCenter[i][k].x;
					in >> ptLN->vCenter[i][k].y;
				}
			}
		}

	} else {
		cerr << "Could not read tree: " << filename << endl;
	}

	in.close();

}


/////////////////////// IO Function /////////////////////////////

bool CRTree::saveTree(const char* filename) const {
	cout << "Save Tree " << filename << endl;

	bool done = false;

	ofstream out(filename);
	if(out.is_open()) {

		out << max_depth << " " << num_leaf << " " << num_cp << endl;

		// save tree nodes
		int* ptT = &treetable[0];
		int depth = 0;
		unsigned int step = 2;
		for(unsigned int n=0; n<num_nodes; ++n) {
			// get depth from node
			if(n==step-1) {
				++depth;
				step *= 2;
			}

			out << n << " " << depth << " ";
			for (unsigned int i = 0; i<NUM_COL; ++i, ++ptT) {
				out << *ptT << " ";
			}
			out << endl;
		}
		out << endl;

		// save tree leafs
		LeafNode* ptLN = &leaf[0];
		for(unsigned int l=0; l<num_leaf; ++l, ++ptLN) {
			out << l << " " << ptLN->pfg << " " << ptLN->vCenter.size() << " ";
			
			for(unsigned int i=0; i<ptLN->vCenter.size(); ++i) {
				for(unsigned int k=0; k<ptLN->vCenter[i].size(); ++k) {
					out << ptLN->vCenter[i][k].x << " " << ptLN->vCenter[i][k].y << " ";
				}
			}
			out << endl;
		}

		out.close();

		done = true;
	}


	return done;
}
/*bool CRTree::loadTree(const char* filename) {

	cout << "Load Tree (BIN) " << filename << " " << flush;
	int dummy;
	bool success = true;

	FILE* fp = fopen(filename, "rb");

	if (!fp){
		cout << "failed" << endl;
		return false;
	}

	success &= (fread(&max_depth, sizeof(int), 1, fp) == 1);
	success &= (fread(&num_leaf, sizeof(int), 1, fp) == 1);
	success &= (fread(&m_pwidth, sizeof(int), 1, fp) == 1);
	success &= (fread(&m_pheight, sizeof(int), 1, fp) == 1);
	success &= (fread(&m_no_chans, sizeof(int), 1, fp) == 1);

	num_nodes = (int)pow(2.0, int(max_depth + 1)) - 1;   // compute number of existing nodes

	treetable = new int[num_nodes * NUM_COL];   // num_nodes x test size: [index, x1,y1,x2,y,2,w1,h1,w2,h2,channel,threshold]
	int* ptT = &treetable[0];                    // get pointer to the tree table

	// get number of leaves from text file
	leaf = new LeafNode[num_leaf];

	// read tree nodes
	for (unsigned int n = 0; n<num_nodes; ++n) {

		success &= (fread(&dummy, sizeof(int), 1, fp) == 1);
		success &= (fread(&dummy, sizeof(int), 1, fp) == 1);

		//read in the test parameters
		for (unsigned int i = 0; i<NUM_COL; ++i, ++ptT){
			success &= (fread(ptT, sizeof(int), 1, fp) == 1);
		}

	}

	// read tree leafs
	LeafNode* ptLN = &leaf[0];
	for (unsigned int l = 0; l<num_leaf; ++l, ++ptLN) {

		ptLN->mean.create(POSE_SIZE, 1, CV_32FC1); ptLN->mean.setTo(0);

		success &= (fread(&dummy, sizeof(int), 1, fp) == 1);
		success &= (fread(&(ptLN->pfg), sizeof(float), 1, fp) == 1);
		success &= (fread(ptLN->mean.data, sizeof(float), POSE_SIZE, fp) == POSE_SIZE);
		success &= (fread(&(ptLN->trace), sizeof(float), 1, fp) == 1);

	}

	fclose(fp);
	std::cout << " done " << endl;

	return success;

}*/
/////////////////////// Training Function /////////////////////////////

// Start grow tree
void CRTree::growTree(const CRPatch& TrData, int samples) {
	// Get ratio positive patches/negative patches
	int pos = 0;
	vector<vector<const PatchFeature*> > TrainSet( TrData.vLPatches.size() );
	for(unsigned int l=0; l<TrainSet.size(); ++l) {
		TrainSet[l].resize(TrData.vLPatches[l].size());
		
		if(l>0) pos += TrainSet[l].size();
		
		for(unsigned int i=0; i<TrainSet[l].size(); ++i) {
			TrainSet[l][i] = &TrData.vLPatches[l][i];
		}
	}

	// Grow tree
	grow(TrainSet, 0, 0, samples, pos / float(TrainSet[0].size()) );
}

// Called by growTree
void CRTree::grow(const vector<vector<const PatchFeature*> >& TrainSet, int node, unsigned int depth, int samples, float pnratio) {

	if(depth<max_depth && TrainSet[1].size()>0) {	

		vector<vector<const PatchFeature*> > SetA;
		vector<vector<const PatchFeature*> > SetB;
		int* test = new int[NUM_COL - 1];

		// Set measure mode for split: 0 - classification, 1 - regression
		unsigned int measure_mode = 0;
		if( float(TrainSet[0].size()) / float(TrainSet[0].size()+TrainSet[1].size()) >= 0.05 && depth < max_depth-2 )
			measure_mode = cvRandInt( cvRNG ) % 2;

		cout << "MeasureMode " << depth << " " << measure_mode << " " << TrainSet[0].size() << " " << TrainSet[1].size() << endl;
	
		// Find optimal test
		if( optimizeTest(SetA, SetB, TrainSet, test, samples, measure_mode) ) {
			// cout << "Channel is:" << test[4] << "; Test Mode (1 - ave 0 - pixel):" << test[NUM_COL - 2] << endl;
			// Store binary test for current node
			int* ptT = &treetable[node*NUM_COL];
			ptT[0] = -1; ++ptT; 
			for (int t = 0; t<NUM_COL - 1; ++t)
				ptT[t] = test[t];

			double countA = 0;
			double countB = 0;
			for(unsigned int l=0; l<TrainSet.size(); ++l) {
				cout << "Final_Split A/B " << l << " " << SetA[l].size() << " " << SetB[l].size() << endl; 
				countA += SetA[l].size(); countB += SetB[l].size();
			}
			for(unsigned int l=0; l<TrainSet.size(); ++l) {
				cout << "Final_SplitA: " << SetA[l].size()/countA << "% "; 
			}
			cout << endl;
			for(unsigned int l=0; l<TrainSet.size(); ++l) {
				cout << "Final_SplitB: " << SetB[l].size()/countB << "% "; 
			}
			cout << endl;

			// Go left
			// If enough patches are left continue growing else stop
			if(SetA[0].size()+SetA[1].size()>min_samples) {
				grow(SetA, 2*node+1, depth+1, samples, pnratio);
			} else {
				makeLeaf(SetA, pnratio, 2*node+1);
			}

			// Go right
			// If enough patches are left continue growing else stop
			if(SetB[0].size()+SetB[1].size()>min_samples) {
				grow(SetB, 2*node+2, depth+1, samples, pnratio);
			} else {
				makeLeaf(SetB, pnratio, 2*node+2);
			}

		} else {

			// Could not find split (only invalid one leave split)
			makeLeaf(TrainSet, pnratio, node);
	
		}	

	} else {

		// Only negative patches are left or maximum depth is reached
		makeLeaf(TrainSet, pnratio, node);
	
	}
}

// Create leaf node from patches 
void CRTree::makeLeaf(const std::vector<std::vector<const PatchFeature*> >& TrainSet, float pnratio, int node) {
	// Get pointer
	treetable[node*NUM_COL] = num_leaf;
	LeafNode* ptL = &leaf[num_leaf];

	// Store data
	ptL->pfg = TrainSet[1].size() / float(pnratio*TrainSet[0].size()+TrainSet[1].size());
	ptL->vCenter.resize( TrainSet[1].size() );
	for(unsigned int i = 0; i<TrainSet[1].size(); ++i) {
		ptL->vCenter[i] = TrainSet[1][i]->center;
	}

	// Increase leaf counter
	++num_leaf;
}

bool CRTree::optimizeTest(vector<vector<const PatchFeature*> >& SetA, vector<vector<const PatchFeature*> >& SetB, const vector<vector<const PatchFeature*> >& TrainSet, int* test, unsigned int iter, unsigned int measure_mode) {
	
	bool found = false;

	// temporary data for split into Set A and Set B
	vector<vector<const PatchFeature*> > tmpA(TrainSet.size());
	vector<vector<const PatchFeature*> > tmpB(TrainSet.size());

	// temporary data for finding best test
	vector<vector<IntIndex> > valSet(TrainSet.size());
	double tmpDist;
	// maximize!!!!
	double bestDist = -DBL_MAX; 
	int* tmpTest = new int[NUM_COL - 1];

	// Find best test of ITER iterations
	for(unsigned int i =0; i<iter; ++i) {

		// reset temporary data for split
		for(unsigned int l =0; l<TrainSet.size(); ++l) {
			tmpA[l].clear();
			tmpB[l].clear(); 
		}

		// generate binary test without threshold
		if (i < TrainSet[1][0]->vPatch.size()){ //first we choose test of patch mean for all channels
			generateTestAve(&tmpTest[0], TrainSet[1][0]->roi.width, TrainSet[1][0]->roi.height, i);
		}
		else{ 
			generateTest(&tmpTest[0], TrainSet[1][0]->roi.width, TrainSet[1][0]->roi.height, TrainSet[1][0]->vPatch.size());
		};
		// compute value for each patch
		evaluateTest(valSet, &tmpTest[0], TrainSet);
		// find min/max values for threshold
		int vmin = INT_MAX;
		int vmax = INT_MIN;
		for(unsigned int l = 0; l<TrainSet.size(); ++l) {
			if(valSet[l].size()>0) {
				if(vmin>valSet[l].front().val)  vmin = valSet[l].front().val;
				if(vmax<valSet[l].back().val )  vmax = valSet[l].back().val;
			}
		}
		int d = vmax-vmin;
		// if it is possible to find a Thershold - find best one
		if(d>0) {

			// Find best threshold - 10 tries!
			for(unsigned int j=0; j<10; ++j) { 

				// Generate some random thresholds
				int tr = (cvRandInt( cvRNG ) % (d)) + vmin; 

				// Split training data into two sets A,B accroding to threshold t 
				split(tmpA, tmpB, TrainSet, valSet, tr);

				// Do not allow empty set split (all patches end up in set A or B)
				if( tmpA[0].size()+tmpA[1].size()>0 && tmpB[0].size()+tmpB[1].size()>0 ) {

					// Measure quality of split with measure_mode 0 - classification, 1 - regression
					tmpDist = measureSet(tmpA, tmpB, measure_mode);

					// Take binary test with best split
					if(tmpDist>bestDist) {

						found = true;
						bestDist = tmpDist;
						// saving [x1,y1,x2,y2,chan]
						for(int t=0; t<5;++t) test[t] = tmpTest[t];
						test[5] = tr; //saving the best thershold to the test *int Arrray
						test[NUM_COL - 2] = tmpTest[NUM_COL - 2];
						SetA = tmpA;
						SetB = tmpB;
					}

				}

			} // end for j

		}

	} // end iter

	// return true if a valid test has been found
	// test is invalid if only splits with an empty set A or B has been created
	return found;
}

void CRTree::evaluateTest(std::vector<std::vector<IntIndex> >& valSet, const int* test, const std::vector<std::vector<const PatchFeature*> >& TrainSet) {
	for(unsigned int l=0;l<TrainSet.size();++l) {
		valSet[l].resize(TrainSet[l].size());
		for(unsigned int i=0;i<TrainSet[l].size();++i) {

			// pointer to channel
			CvMat* ptC = TrainSet[l][i]->vPatch[test[4]];
			switch (test[NUM_COL - 2]){ // last column has the type of test currently only pixel based and average of patch
			case 1: {//patch mean
						int patchSum = 0;
						int numel = 0;
						for (int y = test[1]; y < test[3] + 1; y++){
							for (int x = test[0]; x < test[2] + 1; x++){
								patchSum += (int)*(uchar*)cvPtr2D(ptC, y, x);
								numel++;
							}
						}
						valSet[l][i].val = patchSum /(double) numel;
						break;
			}
			default: { // pixel test
						 // get pixel values 
						 int p1 = (int)*(uchar*)cvPtr2D(ptC, test[1], test[0]);
						 int p2 = (int)*(uchar*)cvPtr2D(ptC, test[3], test[2]);
						 valSet[l][i].val = p1 - p2;
						 break;
			}
			}
			valSet[l][i].index = i;
		}
		sort( valSet[l].begin(), valSet[l].end() );
	}
}

void CRTree::evaluateTestSub(std::vector<std::vector<IntIndex> >& valSet, const int* test, const std::vector<std::vector<const PatchFeature*> >& TrainSet) {
	for (unsigned int l = 0; l<TrainSet.size(); ++l) {
		valSet[l].resize(TrainSet[l].size());
		for (unsigned int i = 0; i<TrainSet[l].size(); ++i) {

			// pointer to channel
			CvMat* ptC = TrainSet[l][i]->vPatch[test[8]]; //TODO - probably need to add a vPatch for nonZeros (see below)
			// a1,b1 is the top-left of two sub-patches and a2,b2 defines the right-bottom
			int xa1 = TrainSet[l][i]->roi.x + test[0];		int xa2 = xa1 + test[4];
			int ya1 = TrainSet[l][i]->roi.y + test[1];		int ya2 = ya1 + test[5];
			int xb1 = TrainSet[l][i]->roi.x + test[2];		int xb2 = xb1 + test[6];
			int yb1 = TrainSet[l][i]->roi.y + test[3];		int yb2 = yb1 + test[7];
			// YD: integral image represenation?
			double mz1 = ((int)*(uchar*)cvPtr2D(ptC,ya1, xa1) +
				(int)*(uchar*)cvPtr2D(ptC,ya2, xa2) -
				(int)*(uchar*)cvPtr2D(ptC,ya2, xa1) -
				(int)*(uchar*)cvPtr2D(ptC,ya1, xa2)) /
				(double)MAX(1, nonZeros.at<double>(ya1, xa1) +
				nonZeros.at<double>(ya2, xa2) -
				nonZeros.at<double>(ya2, xa1) -
				nonZeros.at<double>(ya1, xa2)); // NONZERO in my case should be all ones

			double mz2 = ((int)*(uchar*)cvPtr2D(ptC,yb1, xb1) +
				(int)*(uchar*)cvPtr2D(ptC,yb2, xb2) -
				(int)*(uchar*)cvPtr2D(ptC,yb2, xb1) -
				(int)*(uchar*)cvPtr2D(ptC,yb1, xb2)) /
				(double)MAX(1, nonZeros.at<double>(yb1, xb1) +
				nonZeros.at<double>(yb2, xb2) -
				nonZeros.at<double>(yb2, xb1) -
				nonZeros.at<double>(yb1, xb2));

			//check test
			valSet[l][i].val = (mz1 - mz2);
			valSet[l][i].index = i;
		}
		sort(valSet[l].begin(), valSet[l].end());
	}
}

void CRTree::split(vector<vector<const PatchFeature*> >& SetA, vector<vector<const PatchFeature*> >& SetB, const vector<vector<const PatchFeature*> >& TrainSet, const vector<vector<IntIndex> >& valSet, int t) {
	for(unsigned int l = 0; l<TrainSet.size(); ++l) {
		// search largest value such that val<t 
		vector<IntIndex>::const_iterator it = valSet[l].begin();
		while(it!=valSet[l].end() && it->val<t) {
			++it;
		}

		SetA[l].resize(it-valSet[l].begin());
		SetB[l].resize(TrainSet[l].size()-SetA[l].size());

		it = valSet[l].begin();
		for(unsigned int i=0; i<SetA[l].size(); ++i, ++it) {
			SetA[l][i] = TrainSet[l][it->index];
		}
		
		it = valSet[l].begin()+SetA[l].size();
		for(unsigned int i=0; i<SetB[l].size(); ++i, ++it) {
			SetB[l][i] = TrainSet[l][it->index];
		}

	}
}

double CRTree::distMean(const std::vector<const PatchFeature*>& SetA, const std::vector<const PatchFeature*>& SetB) {
	vector<double> meanAx(num_cp,0);
	vector<double> meanAy(num_cp,0);
	for(vector<const PatchFeature*>::const_iterator it = SetA.begin(); it != SetA.end(); ++it) {
		for(unsigned int c = 0; c<num_cp; ++c) {
			meanAx[c] += (*it)->center[c].x;
			meanAy[c] += (*it)->center[c].y;
		}
	}

	for(unsigned int c = 0; c<num_cp; ++c) {
		meanAx[c] /= (double)SetA.size();
		meanAy[c] /= (double)SetA.size();
	}

	vector<double> distA(num_cp,0);
	for(std::vector<const PatchFeature*>::const_iterator it = SetA.begin(); it != SetA.end(); ++it) {
		for(unsigned int c = 0; c<num_cp; ++c) {
			double tmp = (*it)->center[c].x - meanAx[c];
			distA[c] += tmp*tmp;
			tmp = (*it)->center[c].y - meanAy[c];
			distA[c] += tmp*tmp;
		}
	}

	vector<double> meanBx(num_cp,0);
	vector<double> meanBy(num_cp,0);
	for(vector<const PatchFeature*>::const_iterator it = SetB.begin(); it != SetB.end(); ++it) {
		for(unsigned int c = 0; c<num_cp; ++c) {
			meanBx[c] += (*it)->center[c].x;
			meanBy[c] += (*it)->center[c].y;
		}
	}

	for(unsigned int c = 0; c<num_cp; ++c) {
		meanBx[c] /= (double)SetB.size();
		meanBy[c] /= (double)SetB.size();
	}

	vector<double> distB(num_cp,0);
	for(std::vector<const PatchFeature*>::const_iterator it = SetB.begin(); it != SetB.end(); ++it) {
		for(unsigned int c = 0; c<num_cp; ++c) {
			double tmp = (*it)->center[c].x - meanBx[c];
			distB[c] += tmp*tmp;
			tmp = (*it)->center[c].y - meanBy[c];
			distB[c] += tmp*tmp;
		}
	}

	double minDist = DBL_MAX;

	for(unsigned int c = 0; c<num_cp; ++c) {
		distA[c] += distB[c];
		if(distA[c] < minDist) minDist = distA[c];
	}

	return minDist/double( SetA.size() + SetB.size() ); 
}

double CRTree::InfGain(const vector<vector<const PatchFeature*> >& SetA, const vector<vector<const PatchFeature*> >& SetB) {

	// get size of set A
	double sizeA = 0;
	for(vector<vector<const PatchFeature*> >::const_iterator it = SetA.begin(); it != SetA.end(); ++it) {
		sizeA += it->size();
	}

	// negative entropy: sum_i p_i*log(p_i)
	double n_entropyA = 0;
	for(vector<vector<const PatchFeature*> >::const_iterator it = SetA.begin(); it != SetA.end(); ++it) {
		double p = double( it->size() ) / sizeA;
		if(p>0) n_entropyA += p*log(p); 
	}

	// get size of set B
	double sizeB = 0;
	for(vector<vector<const PatchFeature*> >::const_iterator it = SetB.begin(); it != SetB.end(); ++it) {
		sizeB += it->size();
	}

	// negative entropy: sum_i p_i*log(p_i)
	double n_entropyB = 0;
	for(vector<vector<const PatchFeature*> >::const_iterator it = SetB.begin(); it != SetB.end(); ++it) {
		double p = double( it->size() ) / sizeB;
		if(p>0) n_entropyB += p*log(p); 
	}

	return (sizeA*n_entropyA+sizeB*n_entropyB)/(sizeA+sizeB); 
}

/////////////////////// IO functions /////////////////////////////

void LeafNode::show(int delay, int width, int height) {
	char buffer[200];

	print();

	if(vCenter.size()>0) {
		vector<IplImage*> iShow(vCenter[0].size());
		for(unsigned int c = 0; c<iShow.size(); ++c) {
			iShow[c] = cvCreateImage( cvSize(width,height), IPL_DEPTH_8U , 1 );
			cvSetZero( iShow[c] );
			for(unsigned int i = 0; i<vCenter.size(); ++i) {
				int y = height/2+vCenter[i][c].y;
				int x = width/2+vCenter[i][c].x;

				if(x>=0 && y>=0 && x<width && y<height)
					cvSetReal2D( iShow[c],  y,  x, 255 );
			}
			sprintf_s(buffer,"Leaf%d",c);
			cvNamedWindow(buffer,1);
			cvShowImage(buffer, iShow[c]);
		}
		
		cvWaitKey(delay);
		
		for(unsigned int c = 0; c<iShow.size(); ++c) {
			sprintf_s(buffer,"Leaf%d",c);
			cvDestroyWindow(buffer);
			cvReleaseImage(&iShow[c]);
		}
	}
}

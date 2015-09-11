/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRForestDetector.h"
#include <vector>


using namespace std;


void CRForestDetector::detectColor(IplImage *img, vector<IplImage* >& imgDetect, std::vector<float>& ratios,const char* imfile) {

	// extract features
	vector<IplImage*> vImg;
	//CRPatch::extractFeatureChannels(img, vImg);
	CRPatch::extractFeatureChannelsExtra(img, vImg, imfile);

	// reset output image
	for(int c=0; c<(int)imgDetect.size(); ++c)
		cvSetZero( imgDetect[c] );

	// get pointers to feature channels
	int stepImg;
	uchar** ptFCh     = new uchar*[vImg.size()];
	uchar** ptFCh_row = new uchar*[vImg.size()];
	for(unsigned int c=0; c<vImg.size(); ++c) {
		cvGetRawData( vImg[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]);

	// get pointer to output image
	int stepDet;
	float** ptDet = new float*[imgDetect.size()];
	for(unsigned int c=0; c<imgDetect.size(); ++c)
		cvGetRawData( imgDetect[c], (uchar**)&(ptDet[c]), &stepDet);
	stepDet /= sizeof(ptDet[0][0]);

	int xoffset = width/2;
	int yoffset = height/2;
	
	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset; 

	for(y=0; y<img->height-height; ++y, ++cy) {
		// Get start of row
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];
		cx = xoffset; 
		
		for(x=0; x<img->width-width; ++x, ++cx) {					

			// regression for a single patch
			vector<const LeafNode*> result;
			crForest->regression(result, ptFCh_row, stepImg);
			
			// vote for all trees (leafs) 
			for(vector<const LeafNode*>::const_iterator itL = result.begin(); itL!=result.end(); ++itL) {

				// To speed up the voting, one can vote only for patches 
			        // with a probability for foreground > 0.5
			        // 
				//if((*itL)->pfg>0.5) {

					// voting weight for leaf 
					float w = (*itL)->pfg / float( (*itL)->vCenter.size() * result.size() );

					// vote for all points stored in the leaf
					for(vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it!=(*itL)->vCenter.end(); ++it) {

						for(int c=0; c<(int)imgDetect.size(); ++c) {
						  int x = int(cx - (*it)[0].x * ratios[c] + 0.5);
						  int y = cy-(*it)[0].y;
						  if(y>=0 && y<imgDetect[c]->height && x>=0 && x<imgDetect[c]->width) {
						    *(ptDet[c]+x+y*stepDet) += w;
						  }
						}
					}

				// } // end if

			}

			// increase pointer - x
			for(unsigned int c=0; c<vImg.size(); ++c)
				++ptFCh_row[c];

		} // end for x

		// increase pointer - y
		for(unsigned int c=0; c<vImg.size(); ++c)
			ptFCh[c] += stepImg;

	} // end for y 	

	// smooth result image
	for(int c=0; c<(int)imgDetect.size(); ++c)
		cvSmooth( imgDetect[c], imgDetect[c], CV_GAUSSIAN, 3);

	// release feature channels
	for(unsigned int c=0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);
	
	delete[] ptFCh;
	delete[] ptFCh_row;
	delete[] ptDet;

}

void CRForestDetector::detectColorcascade(PatchFeature& p, priority_queue<PatchFeature, vector<PatchFeature>, LessThanFeature>& pos_bad_examples,
	priority_queue<PatchFeature, vector<PatchFeature>, LessThanFeature>& neg_bad_examples, int k, std::vector<float>& ratios) {

	// get pointers to feature channels
	int stepImg;
	uchar** ptFCh = new uchar*[p.vPatch.size()];
	uchar** ptFCh_row = new uchar*[p.vPatch.size()];
	for (unsigned int c = 0; c<p.vPatch.size(); ++c) {
		cvGetRawData(p.vPatch[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]);

	//GT offset from the peak of the distribution
	int xoffset = p.center[0].x;
	int yoffset = p.center[0].y;

	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset;
	cx = xoffset;
	for (unsigned int c = 0; c<p.vPatch.size(); ++c)
		ptFCh_row[c] = &ptFCh[c][0];
	// regression for a single patch
	vector<const LeafNode*> result;
	crForest->regression(result, ptFCh_row, stepImg);
	float se = 0; //Square error of current patch on all availabe trees
	float cmean = 0;
	int numpt = 0;
	// vote for all trees (leafs) itL == iterator for Leafs
	for (vector<const LeafNode*>::const_iterator itL = result.begin(); itL != result.end(); ++itL) {
		// To speed up the voting, one can vote only for patches 
		// with a probability for foreground > 0.5
		// 
		//if((*itL)->pfg>0.5) {
		
		// voting weight for leaf 
		float w = (*itL)->pfg / float((*itL)->vCenter.size() * result.size());
		cmean += w;
		// vote for all points stored in the leaf it == iterator for all points in leaf
		for (vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it != (*itL)->vCenter.end(); ++it) {
			for (int c = 0; c<(int)ratios.size(); ++c) { //for all scales
				int dx = int(cx - (*it)[0].x * ratios[c] + 0.5);
				int dy = int(cy - (*it)[0].y * ratios[c] + 0.5);
				se += w*(dx ^ 2 + dy ^ 2); // aggregating the square error of this patch on all trees
				numpt++;
			}
		}
		// } // end if
	}
	double rmse = sqrt(se / numpt); // normalizing the square error with n to get MSE and retreive the root to get RMSE
	//if heap is either full or current patch is worst than all other patches in the heap take out the lower push it in
	p.err = rmse;
	p.cmean = cmean;
	if (!p.fg){ //negative example
		if (neg_bad_examples.size() < k || neg_bad_examples.top().cmean<p.cmean){
			if (neg_bad_examples.size() == k) neg_bad_examples.pop();
			neg_bad_examples.push(p);
		}
	}
	else{ //positive example
		if (pos_bad_examples.size() < k || pos_bad_examples.top().err<p.err){
			if (pos_bad_examples.size() == k) pos_bad_examples.pop();
				pos_bad_examples.push(p);
			}
		}
	// increase pointer - x
	for (unsigned int c = 0; c<p.vPatch.size(); ++c)
		++ptFCh_row[c];
	// increase pointer - y
	for (unsigned int c = 0; c<p.vPatch.size(); ++c)
		ptFCh[c] += stepImg;
	// release feature channels
	delete[] ptFCh;
	delete[] ptFCh_row;
}

void CRForestDetector::detectPyramid(IplImage *img, vector<vector<IplImage*> >& vImgDetect, std::vector<float>& ratios,const char* imfile) {	

	if(img->nChannels==1) {

		std::cerr << "Gray color images are not supported." << std::endl;

	} else { // color

		cout << "Timer" << endl;
		int tstart = clock();

		for(int i=0; i<int(vImgDetect.size()); ++i) {
			IplImage* cLevel = cvCreateImage( cvSize(vImgDetect[i][0]->width,vImgDetect[i][0]->height) , IPL_DEPTH_8U , 3);				
			cvResize( img, cLevel, CV_INTER_LINEAR );	

			// detection
			detectColor(cLevel,vImgDetect[i],ratios,imfile);

			cvReleaseImage(&cLevel);
		}

		cout << "Time " << (double)(clock() - tstart)/CLOCKS_PER_SEC << " sec" << endl;

	}

}

void CRForestDetector::detectPyramidcascade(PatchFeature& p, priority_queue<PatchFeature, vector<PatchFeature>, LessThanFeature>& pos_bad_examples,priority_queue<PatchFeature, vector<PatchFeature>, LessThanFeature>& neg_bad_examples, int k, std::vector<float>& ratios) {

	cout << "Timer" << endl;
	int tstart = clock();
	// if you'd like a ratio insert you'll have to figure out how to resize vPatches of PatchFeature
	//for (int i = 0; i<int(ratios.size()); ++i) {
		// detection
		detectColorcascade(p, pos_bad_examples, neg_bad_examples, k, ratios);
	//}

	cout << "Time " << (double)(clock() - tstart) / CLOCKS_PER_SEC << " sec" << endl;

}







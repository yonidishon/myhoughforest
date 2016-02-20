/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRForestDetector.h"
#include <vector>


using namespace std;

// DETECTORS
void CRForestDetector::detectColor(IplImage *img, vector<IplImage* >& imgDetect, std::vector<float>& ratios,const char* imfile) {

	// extract features
	vector<IplImage*> vImg;
	//CRPatch::extractFeatureChannels(img, vImg);
	//CRPatch::extractFeatureChannelsExtra(img, vImg, imfile); //vImg are features map size(Img)
	CRPatch::extractPCAChannels(img, vImg, imfile); //vImg are features map size(Img)

	// reset output image
	for(int c=0; c<(int)imgDetect.size(); ++c)
		cvSetZero( imgDetect[c] );

	// get pointers to feature channels
	int stepImg; //size in pixels of one row of Img
	uchar** ptFCh     = new uchar*[vImg.size()];
	uchar** ptFCh_row = new uchar*[vImg.size()];
	for(unsigned int c=0; c<vImg.size(); ++c) {
		cvGetRawData( vImg[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]); // now stepImg is the number of pixels in one row of the img

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
			ptFCh_row[c] = &ptFCh[c][0]; //ptFCh_row is a pointer to the 1st element in the row of each channel
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
				if((*itL)->pfg>=0.8) {

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

				} // end if

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

float CRForestDetector::detectColorHardNeg(IplImage* img, priority_queue<PatchHardMining, vector<PatchHardMining>, LessThanPatchHardMining>& pos_bad_examples,
	vector<float>& ratios, const char* imfile, const char* filename, CvRect vBBox, CvPoint vCenter, int max_neg_samples) {
	//imfile is the full path to the image *img
	float se = 0; //Square error of the entire training image
	uint numpatches = 0;//number of patches in image
	// extract features
	vector<IplImage*> vImg;
	//CRPatch::extractFeatureChannels(img, vImg);
	CRPatch::extractFeatureChannelsExtra(img, vImg, imfile);
	int neg_samples_counter = 0;
	// get pointers to feature channels
	int stepImg;
	uchar** ptFCh = new uchar*[vImg.size()];
	uchar** ptFCh_row = new uchar*[vImg.size()];
	for (unsigned int c = 0; c<vImg.size(); ++c) {
		cvGetRawData(vImg[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]);

	int xoffset = width/2;
	int yoffset = height/2;

	int x, y, cx, cy; // x,y top left; cx,cy center of patch
	cy = yoffset;

	for (y = 0; y< (img->height - height); ++y, ++cy) {
		// Get start of row
		for (unsigned int c = 0; c<vImg.size(); ++c)
			ptFCh_row[c] = &ptFCh[c][0];
		cx = xoffset;

		for (x = 0; x < img->width - width; ++x, ++cx) {
			numpatches++;

			// regression for a single patch
			vector<const LeafNode*> result;
			crForest->regression(result, ptFCh_row, stepImg);
			float cmean = 0; //"Class Mean": Probability of being foreground according to the leaf arrived to.
			int numpt = 0; // number of points in a specific leaf
			// vote for all trees (leafs) *itL is a leaf <itL>
			for (vector<const LeafNode*>::const_iterator itL = result.begin(); itL != result.end(); ++itL) {

				// To speed up the voting, one can vote only for patches 
				// with a probability for foreground > 0.5
				// 
				//if((*itL)->pfg>0.5) {

				// voting weight for each point in leaf  (vCenter.Size is number of training patches got to this leaft)
				//							(result.size is the number of trees in the forest
				cmean += (*itL)->pfg * (float((*itL)->vCenter.size()));
				numpt += (*itL)->vCenter.size();
				// vote for all points stored in the leaf ->  *it is a point <it> in the leaf 
				// } // end if
				// voting weight for each point in leaf 
				float w = (*itL)->pfg / float((*itL)->vCenter.size() * result.size());

				// vote for all points stored in the leaf
				for (vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it != (*itL)->vCenter.end(); ++it) {

					for (int c = 0; c<(int)ratios.size(); ++c) {
						//int x = int(cx - (*it)[0].x * ratios[c] + 0.5);
						//int y = cy - (*it)[0].y;
						int dx = int(vCenter.x - (x+xoffset-(*it)[0].x) * ratios[c] + 0.5);
						int dy = int(vCenter.y - (y+yoffset-(*it)[0].y) * ratios[c] + 0.5);
						//cout << (dx <10 && -dx>-10) || (dy<5 && -dy > -5);
						//cout << " dx is: " << dx << " dy is: " << dy << endl;
						//cout << "Weighted Square error: " << w*(pow(dx, 2) + pow(dy, 2)) << endl;
						se += w*(pow(dx, 2) + pow(dy, 2)); // aggregating the square error of this patch on all trees
					}
				}
			}
			cmean = cmean / float(numpt); //average class decision calculated as a weighted average between trees (with points arrived to each leaf as a bais)
			//if heap is either full or current patch is worst than all other patches in the heap take out the lower push it in
			if (!isinposrect(CvRect(vBBox.x, vBBox.y, vBBox.width, vBBox.height),CvPoint(x, y))){// condition for patch to be negative 100% contained in BB
				if (pos_bad_examples.size() < max_neg_samples || pos_bad_examples.top().cmean<cmean){
					if (pos_bad_examples.size() == max_neg_samples) {
						////sanaty check - remove
						//while (!pos_bad_examples.empty()){
						//	cout << "top of heap is: " << pos_bad_examples.top().cmean << endl;
						//	pos_bad_examples.pop();
						//}
						//cout << "(" << x << "," << y << ") ";
						//cout << "smallest in heap was (now popping): " << pos_bad_examples.top().cmean << endl;
						pos_bad_examples.pop();
					}
					char buffer[100];
					sprintf(buffer, "%s %i %i %i %i", filename, x, y, x + width, y + height);
					PatchHardMining* neg_patch = new PatchHardMining();
					neg_patch->patchpath = string(buffer);
					neg_patch->cmean = cmean;
					pos_bad_examples.push(*neg_patch);
				}

			}
			// increase pointer - x
			for (unsigned int c = 0; c<vImg.size(); ++c)
				++ptFCh_row[c];

		} // end for x

		// increase pointer - y
		for (unsigned int c = 0; c<vImg.size(); ++c)
			ptFCh[c] += stepImg;

	} // end for y 	

	// release feature channels
	for (unsigned int c = 0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);

	delete[] ptFCh;
	delete[] ptFCh_row;
	return se / numpatches;
}

void CRForestDetector::detectColorcascade(PatchFeature& p, priority_queue<PatchFeature, vector<PatchFeature>, LessThanFeature>& pos_bad_examples,
	priority_queue<PatchFeature, vector<PatchFeature>, LessThanFeature>& neg_bad_examples, int k, std::vector<float>& ratios) {

	// get pointers to feature channels
	int stepImg;
	uchar** ptFCh = new uchar*[p.vPatch.size()];
	uchar** ptFCh_row = new uchar*[p.vPatch.size()];
	for (unsigned int c = 0; c < p.vPatch.size(); ++c) {
		cvGetRawData(p.vPatch[c], (uchar**)&(ptFCh[c]), &stepImg);
	}
	stepImg /= sizeof(ptFCh[0][0]);

	//GT offset from the peak of the distribution
	int x, y, cx, cy;// x,y top left; 
	if (p.fg){
		//cx, cy center of patch
		cx = p.center[0].x;
		cy = p.center[0].y;
	}

	for (unsigned int c = 0; c<p.vPatch.size(); ++c)
		ptFCh_row[c] = &ptFCh[c][0];
	// regression for a single patch
	vector<const LeafNode*> result;
	crForest->regression(result, ptFCh_row, stepImg);
	float se = 0; //Square error of current patch on all availabe trees
	float cmean = 0; //"Class Mean": Probability of being foreground according to the leaf arrived to.
	int numpt = 0; // number of points in a specific leaf
	// vote for all trees (leafs) itL == iterator for Leafs
	for (vector<const LeafNode*>::const_iterator itL = result.begin(); itL != result.end(); ++itL) {
		// To speed up the voting, one can vote only for patches 
		// with a probability for foreground > 0.5
		// 
		//if((*itL)->pfg>0.5) {
		
		// voting weight for each point in current leaf 
		float w = (*itL)->pfg / float((*itL)->vCenter.size() * result.size());

		cmean += ((*itL)->pfg)*(float((*itL)->vCenter.size()));
		numpt += ((*itL)->vCenter.size());
		if (p.fg){
			// vote for all points stored in the leaf it == iterator for all points in leaf
			for (vector<vector<CvPoint> >::const_iterator it = (*itL)->vCenter.begin(); it != (*itL)->vCenter.end(); ++it) {
				for (int c = 0; c < (int)ratios.size(); ++c) { //for all scales
					int dx = int(cx - (*it)[0].x * ratios[c] + 0.5);
					int dy = int(cy - (*it)[0].y * ratios[c] + 0.5);
					// cout << "dx is: " << dx << " dy is: " << dy << endl;
					// cout << "Weighted Square error: " << w*(pow(dx, 2) + pow(dy, 2)) << endl;
					se += w*(pow(dx,2) + pow(dy,2)); // aggregating the square error of this patch on all trees
				}
			}
		}
		// } // end if
	}
	float rmse = sqrt(se / float(numpt)); // normalizing the square error with n to get MSE and retreive the root to get RMSE
	cmean = cmean / float(numpt); //average class decision calculated as a weighted average between trees (with points arrived to each leaf as a bais)
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

// PYRAMIDS
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

float CRForestDetector::detectPyramidhard(IplImage *img, std::vector<float>& scales, std::vector<float>& ratios, const char *imfile, priority_queue<PatchHardMining, vector<PatchHardMining>, LessThanPatchHardMining>& neg_examples, const char *filename, CvRect vBBox, CvPoint vCenter, int max_neg_samples) {
	float mse = 0;
	if (img->nChannels == 1) {

		std::cerr << "gray color images are not supported." << std::endl;

	}
	else { // color

		cout << "Timer" << endl;
		int tstart = clock();

		for (int i = 0; i<int(scales.size()); ++i) {
			//mockup image just for the resize
			IplImage* cLevel = cvCreateImage(cvSize(int(img->width*scales[i] + 0.5), int(img->height*scales[i] + 0.5)), IPL_DEPTH_8U, 3);
			cvResize(img, cLevel, CV_INTER_LINEAR);

			// detection
			mse += detectColorHardNeg(img, neg_examples, ratios, imfile, filename, vBBox, vCenter, max_neg_samples); // TODO - if multi scale need to change this func

			cvReleaseImage(&cLevel);
		}

		cout << "Time " << (double)(clock() - tstart) / CLOCKS_PER_SEC << " sec" << endl;

	}
	return mse;
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
// HELPERS
bool CRForestDetector::isinposrect(CvRect& BB, CvPoint& l_cor){
	cv::Rect2d rect(BB.x, BB.y, BB.width - width, BB.height - height);//smaller BB to check if the left point of patch is contained within (positive patch)
	if (rect.contains(l_cor)) return true;
	return false;
}







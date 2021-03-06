/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
 
// You may use, copy, reproduce, and distribute this Software for any 
// non-commercial purpose, subject to the restrictions of the 
// Microsoft Research Shared Source license agreement ("MSR-SSLA"). 
// Some purposes which can be non-commercial are teaching, academic 
// research, public demonstrations and personal experimentation. You 
// may also distribute this Software with books or other teaching 
// materials, or publish the Software on websites, that are intended 
// to teach the use of the Software for academic or other 
// non-commercial purposes.
// You may not use or distribute this Software or any derivative works 
// in any form for commercial purposes. Examples of commercial 
// purposes would be running business operations, licensing, leasing, 
// or selling the Software, distributing the Software for use with 
// commercial products, using the Software in the creation or use of 
// commercial products or any other activity which purpose is to 
// procure a commercial gain to you or others.
// If the Software includes source code or data, you may create 
// derivative works of such portions of the Software and distribute 
// the modified Software for non-commercial purposes, as provided 
// herein.

// THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO 
// EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT 
// LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A 
// PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR 
// ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR 
// NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL 
// FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST 
// PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR 
// DERIVATIVE WORKS.

// NEITHER MICROSOFT NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE 
// LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS MSR-SSLA, 
// INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL 
// DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT 
// LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF 
// LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE 
// WORKS.

// When using this software, please acknowledge the effort that 
// went into development by referencing the paper:
//
// Gall J. and Lempitsky V., Class-Specific Hough Forests for 
// Object Detection, IEEE Conference on Computer Vision and Pattern 
// Recognition (CVPR'09), 2009.

// Note that this is not the original software that was used for 
// the paper mentioned above. It is a re-implementation for Linux. 

*/


#define PATH_SEP "\\"

#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <Windows.h>
#include "Shlwapi.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "CRForestDetector.h"

using namespace std;
using namespace cv;

// Path to trees
string treepath;
// Number of trees
int ntrees;
// Patch width
int p_width;
// Patch height
int p_height;
// Path to images
string impath;
// File with names of images
string imfiles;
// Extract features
bool xtrFeature;
// Scales
vector<float> scales;
// Ratio
vector<float> ratios;
// Output path
string outpath;
// scale factor for output image (default: 128)
int out_scale;
// Path to positive examples
string trainpospath;
// File with postive examples
string trainposfiles;
// Subset of positive images -1: all images
int subsamples_pos;
// Sample patches from pos. examples
unsigned int samples_pos; 
// Path to positive examples
string trainnegpath;
// File with postive examples
string trainnegfiles;
// Subset of neg images -1: all images
int subsamples_neg;
// Samples from pos. examples
unsigned int samples_neg;

// offset for saving tree number
int off_tree;

BOOL DirectoryExists(LPCTSTR szPath)
{
	DWORD dwAttrib = GetFileAttributes(szPath);

	return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
		(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

// load config file for dataset
void loadConfig(const char* filename, int mode) {
	char buffer[400];
	ifstream in(filename);

	if(in.is_open()) {

		// Path to trees
		in.getline(buffer,400);
		in.getline(buffer,400); 
		treepath = buffer;
		// Number of trees
		in.getline(buffer,400); 
		in >> ntrees;
		in.getline(buffer,400); 
		// Patch width
		in.getline(buffer,400); 
		in >> p_width;
		in.getline(buffer,400); 
		// Patch height
		in.getline(buffer,400); 
		in >> p_height;
		in.getline(buffer,400); 
		// Path to images
		in.getline(buffer,400); 
		in.getline(buffer,400); 
		impath = buffer;
		// File with names of images
		in.getline(buffer,400);
		in.getline(buffer,400);
		imfiles = buffer;
		// Extract features
		in.getline(buffer,400);
		in >> xtrFeature;
		in.getline(buffer,400); 
		// Scales
		in.getline(buffer,400);
		int size;
		in >> size;
		scales.resize(size);
		for(int i=0;i<size;++i)
			in >> scales[i];
		in.getline(buffer,400); 
		// Ratio
		in.getline(buffer,400);
		in >> size;
		ratios.resize(size);
		for(int i=0;i<size;++i)
			in >> ratios[i];
		in.getline(buffer,400); 
		// Output path
		in.getline(buffer,400);
		in.getline(buffer,400);
		outpath = buffer;
		// Scale factor for output image (default: 128)
		in.getline(buffer,400);
		in >> out_scale;
		in.getline(buffer,400);
		// Path to positive examples
		in.getline(buffer,400);
		in.getline(buffer,400);
		trainpospath = buffer;
		// File with postive examples
		in.getline(buffer,400);
		in.getline(buffer,400);
		trainposfiles = buffer;
		// Subset of positive images -1: all images
		in.getline(buffer,400);
		in >> subsamples_pos;
		in.getline(buffer,400);
		// Samples from pos. examples
		in.getline(buffer,400);
		in >> samples_pos;
		in.getline(buffer,400);
		// Path to positive examples
		in.getline(buffer,400);
		in.getline(buffer,400);
		trainnegpath = buffer;
		// File with postive examples
		in.getline(buffer,400);
		in.getline(buffer,400);
		trainnegfiles = buffer;
		// Subset of negative images -1: all images
		in.getline(buffer,400);
		in >> subsamples_neg;
		in.getline(buffer,400);
		// Samples from pos. examples
		in.getline(buffer,400);
		in >> samples_neg;
		//in.getline(buffer,400);

	} else {
		cerr << "File not found " << filename << endl;
		exit(-1);
	}
	in.close();

	switch ( mode ) { 
		case 0:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Training:         " << endl;
		cout << "Patches:          " << p_width << " " << p_height << endl;
		cout << "Train pos:        " << trainpospath << endl;
		cout << "                  " << trainposfiles << endl;
		cout << "                  " << subsamples_pos << " " << samples_pos << endl;
		cout << "Train neg:        " << trainnegpath << endl;
		cout << "                  " << trainnegfiles << endl;
		cout << "                  " << subsamples_neg << " " << samples_neg << endl;
		cout << "Trees:            " << ntrees << " " << off_tree << " " << treepath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;

		case 1:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Show:             " << endl;
		cout << "Trees:            " << ntrees << " " << treepath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;

		default:
		cout << endl << "------------------------------------" << endl << endl;
		cout << "Detection:        " << endl;
		cout << "Trees:            " << ntrees << " " << treepath << endl;
		cout << "Patches:          " << p_width << " " << p_height << endl;
		cout << "Images:           " << impath << endl;
		cout << "                  " << imfiles << endl;
		cout << "Scales:           "; for(unsigned int i=0;i<scales.size();++i) cout << scales[i] << " "; cout << endl;
		cout << "Ratios:           "; for(unsigned int i=0;i<ratios.size();++i) cout << ratios[i] << " "; cout << endl;
		cout << "Extract Features: " << xtrFeature << endl;
		cout << "Output:           " << out_scale << " " << outpath << endl;
		cout << endl << "------------------------------------" << endl << endl;
		break;
	}

}

// load test image filenames
void loadImFile(std::vector<string>& vFilenames) {
	
	char buffer[400];

	ifstream in(imfiles.c_str());
	if(in.is_open()) {

		unsigned int size;
		in >> size; //size = 10;
		in.getline(buffer,400); 
		vFilenames.resize(size);

		for(unsigned int i=0; i<size; ++i) {
			in.getline(buffer,400);      
			vFilenames[i] = buffer;	
		}

	} else {
		cerr << "File not found " << imfiles.c_str() << endl;
		exit(-1);
	}

	in.close();
}

// load positive training image filenames
void loadTrainPosFile(std::vector<string>& vFilenames, std::vector<CvRect>& vBBox, std::vector<std::vector<CvPoint> >& vCenter) {

	unsigned int size, numop; 
	ifstream in(trainposfiles.c_str());

	if(in.is_open()) {
		in >> size;
		in >> numop;
		cout << "Load Train Pos Examples: " << size << " - " << numop << endl;

		vFilenames.resize(size);
		vCenter.resize(size);
		vBBox.resize(size);

		for(unsigned int i=0; i<size; ++i) {
			// Read filename
			in >> vFilenames[i];

			// Read bounding box
			in >> vBBox[i].x; in >> vBBox[i].y; 
			in >> vBBox[i].width;
			vBBox[i].width -= vBBox[i].x; 
			in >> vBBox[i].height;
			vBBox[i].height -= vBBox[i].y;

			if(vBBox[i].width<p_width || vBBox[i].height<p_height) {
			  cout << "Width or height are too small" << endl; 
			  cout << vFilenames[i] << endl;
			  exit(-1); 
			}

			// Read center points
			vCenter[i].resize(numop);
			for(unsigned int c=0; c<numop; ++c) {			
				in >> vCenter[i][c].x;
				in >> vCenter[i][c].y;
			}				
		}

		in.close();
	} else {
		cerr << "File not found " << trainposfiles.c_str() << endl;
		exit(-1);
	}
}

// load negative training image filenames
void loadTrainNegFile(std::vector<string>& vFilenames, std::vector<CvRect>& vBBox) {

	unsigned int size, numop; 
	ifstream in(trainnegfiles.c_str());

	if(in.is_open()) {
		in >> size;
		in >> numop;
		cout << "Load Train Neg Examples: " << size << " - " << numop << endl;

		vFilenames.resize(size);
		if(numop>0)
			vBBox.resize(size);
		else
			vBBox.clear();

		for(unsigned int i=0; i<size; ++i) {
			// Read filename
			in >> vFilenames[i];

			// Read bounding box (if available)
			if(numop>0) {
				in >> vBBox[i].x; in >> vBBox[i].y; 
				in >> vBBox[i].width;
				vBBox[i].width -= vBBox[i].x; 
				in >> vBBox[i].height;
				vBBox[i].height -= vBBox[i].y;

				if(vBBox[i].width<p_width || vBBox[i].height<p_height) {
				  cout << "Width or height are too small" << endl; 
				  cout << vFilenames[i] << endl;
				  exit(-1); 
				}

			}

				
		}

		in.close();
	} else {
		cerr << "File not found " << trainposfiles.c_str() << endl;
		exit(-1);
	}
}

// Show leaves
void show() {
	// Init forest with number of trees
	CRForest crForest( ntrees ); 

	// Load forest
	crForest.loadForest(treepath.c_str());	

	// Show leaves
	crForest.show(100,100);
}

// Run detector
void detect(CRForestDetector& crDetect) {
	

	// Load image names
	vector<string> vFilenames;
	loadImFile(vFilenames);
				
	char buffer[200];

	// Storage for output
	vector<vector<IplImage*> > vImgDetect(scales.size());	

	// Run detector for each image
	for(unsigned int i=0; i<vFilenames.size(); ++i) {

		// Load image
		IplImage *img = 0;
		img = cvLoadImage(vFilenames[i].c_str(),CV_LOAD_IMAGE_COLOR);
		if(!img) {
			cout << "Could not load image file: " << (impath + "/" + vFilenames[i]).c_str() << endl;
			exit(-1);
		}	

		// Prepare scales
		for(unsigned int k=0;k<vImgDetect.size(); ++k) {
			vImgDetect[k].resize(ratios.size());
			for(unsigned int c=0;c<vImgDetect[k].size(); ++c) {
				vImgDetect[k][c] = cvCreateImage( cvSize(int(img->width*scales[k]+0.5),int(img->height*scales[k]+0.5)), IPL_DEPTH_32F, 1 );
			}
		}

		// Detection for all scales
		crDetect.detectPyramid(img, vImgDetect, ratios, vFilenames[i].c_str());

		// Store result
		string delimiter = ".";
		string s = vFilenames[i].c_str(); 
		string token = s.substr(0, s.find(delimiter)); //fullpath without file extention
		size_t found = token.find_last_of("/\\");
		string fname = token.substr(found + 1);//filename
		string path = token.substr(0,found); //full path of image without filename
		string pfolder = path.substr(path.find_last_of("/\\'") + 1); //parent folder only
		string curfolder = outpath + "\\" + pfolder; //store path for detection
		// Check if folder for result is exist and create if not
		if (!DirectoryExists(curfolder.c_str())){
			string execstr1 = "mkdir ";
			execstr1 += curfolder;
			system(execstr1.c_str());
		}	
		
		for(unsigned int k=0;k<vImgDetect.size(); ++k) {
			IplImage* tmp = cvCreateImage(cvSize(vImgDetect[k][0]->width, vImgDetect[k][0]->height), IPL_DEPTH_16U, 1);
			for(unsigned int c=0;c<vImgDetect[k].size(); ++c) {
				//cvConvertScale( vImgDetect[k][c], tmp, out_scale); //80 128
				cvConvertScale(vImgDetect[k][c], tmp, out_scale);
				sprintf_s(buffer, "%s\\%s_sc%d_c%d_predmap.png", curfolder.c_str(), fname.c_str(), k, c);
				cvSaveImage(buffer, tmp);
				//cvSaveImage(buffer, vImgDetect[k][c]);
				cvReleaseImage(&vImgDetect[k][c]);
			}
			cvReleaseImage(&tmp);
		}

		// Release image
		cvReleaseImage(&img);

	}

}
	
// Extract patches from training data
void extract_Patches(CRPatch& Train, CvRNG* pRNG) {
		
	vector<string> vFilenames;
	vector<CvRect> vBBox;
	vector<vector<CvPoint> > vCenter;

	 //load positive file list
	loadTrainPosFile(vFilenames,  vBBox, vCenter);

	// load postive images and extract patches
	for(int i=0; i<(int)vFilenames.size(); ++i) {

	  if(i%50==0) cout << i << " " << flush;

	  if(subsamples_pos <= 0 || (int)vFilenames.size()<=subsamples_pos || (cvRandReal(pRNG)*double(vFilenames.size()) < double(subsamples_pos)) ) {

			// Load image
			IplImage *img = 0;
			//img = cvLoadImage((trainpospath + "/" + vFilenames[i]).c_str(),CV_LOAD_IMAGE_COLOR);
			img = cvLoadImage(vFilenames[i].c_str(), CV_LOAD_IMAGE_COLOR);
			if(!img) {
				//cout << "Could not load image file: " << (trainpospath + "/" + vFilenames[i]).c_str() << endl;
				cout << "Could not load image file: " << vFilenames[i].c_str() << endl;
				exit(-1);
			}	

			// Extract positive training patches
			Train.extractPatches(img, vFilenames[i].c_str(), samples_pos, 1, &vBBox[i], &vCenter[i]);

			// Release image
			cvReleaseImage(&img);

	  }
			
	}
	cout << endl;

	// load negative file list
	loadTrainNegFile(vFilenames,  vBBox);

	// load negative images and extract patches
	for(int i=0; i<(int)vFilenames.size(); ++i) {

		if(i%50==0) cout << i << " " << flush;

		if(subsamples_neg <= 0 || (int)vFilenames.size()<=subsamples_neg || ( cvRandReal(pRNG)*double(vFilenames.size()) < double(subsamples_neg) ) ) {

			// Load image
			IplImage *img = 0;
			img = cvLoadImage(vFilenames[i].c_str(),CV_LOAD_IMAGE_COLOR);

			if(!img) {
				cout << "Could not load image file: " << (trainnegpath + "/" + vFilenames[i]).c_str() << endl;
				exit(-1);
			}	

			// Extract negative training patches
			if(vBBox.size()==vFilenames.size())
				Train.extractPatches(img, vFilenames[i].c_str(), samples_neg, 0, &vBBox[i]);
			else
				Train.extractPatches(img, vFilenames[i].c_str(), samples_neg, 0);

			// Release image
			cvReleaseImage(&img);

		}
			
	}
	cout << endl;
}

void nonMaximaSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask) {

	// initialise the block mask and destination
	const int M = src.rows;
	const int N = src.cols;
	const bool masked = !mask.empty();
	Mat block = 255*Mat_<uint8_t>::ones(Size(2*sz+1,2*sz+1));
	dst = Mat_<uint8_t>::zeros(src.size());

	// iterate over image blocks
	for (int m = 0; m < M; m+=sz+1) {
		for (int n = 0; n < N; n+=sz+1) {
			Point  ijmax;
			double vcmax, vnmax;
			//if (m == 110 && n == 198) {
			//	printf("Debug\n");
			//}
			//printf("row %d and col %d\n", m, n);
			// get the maximal candidate within the block
			Range ic(m, min(m+sz+1,M));
			Range jc(n, min(n+sz+1,N));
			minMaxLoc(src(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic,jc) : noArray());
			Point cc = ijmax + Point(jc.start,ic.start);

			// search the neighbours centered around the candidate for the true maxima
			Range in(max(cc.y-sz,0), min(cc.y+sz+1,M));
			Range jn(max(cc.x-sz,0), min(cc.x+sz+1,N));

			// mask out the block whose maxima we already know
			Mat_<uint8_t> blockmask;
			block(Range(0,in.size()), Range(0,jn.size())).copyTo(blockmask);
			Range iis(ic.start-in.start, min(ic.start-in.start+sz+1, in.size()));
			Range jis(jc.start-jn.start, min(jc.start-jn.start+sz+1, jn.size()));
			blockmask(iis, jis) = Mat_<uint8_t>::zeros(Size(jis.size(),iis.size()));

			minMaxLoc(src(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in,jn).mul(blockmask) : blockmask);
			Point cn = ijmax + Point(jn.start, in.start);

			// if the block centre is also the neighbour centre, then it's a local maxima
			if (vcmax > vnmax) {
				dst.at<uint8_t>(cc.y, cc.x) = 255;
			}
		}
	}
}

static void meshgrid(const Mat &xgv, const Mat &ygv,
	Mat1i &X, Mat1i &Y)
{
	repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

// helper function (maybe that goes somehow easier)
static void meshgridTest(const Range &xgv, const Range &ygv,
	Mat1i &X, Mat1i &Y)
{
	vector<int> t_x, t_y;
	for (int i = xgv.start; i < xgv.end; i++) t_x.push_back(i);
	for (int i = ygv.start; i < ygv.end; i++) t_y.push_back(i);
	meshgrid(Mat(t_x), Mat(t_y), X, Y);
}

Mat nmsMat2GMM(Mat& nmsmat,Mat& img,int sigma=10){
	vector<Point> locations;   // output, locations of non-zero pixels 
	findNonZero(nmsmat, locations);
	vector<float> pnts;
	vector<Point> ps;
	for (int i = 0; i < locations.size(); i+=10){
		ps.push_back(locations[i]);
		pnts.push_back(img.at<float>(locations[i].y, locations[i].x));
	}
	Mat1i X, Y;
	meshgridTest(Range(0, nmsmat.cols), Range(0, nmsmat.rows), X, Y);
	Mat GMM(nmsmat.rows, nmsmat.cols, CV_64F, Scalar::all(0));
	for (int i = 0; i < ps.size(); i++){
		Mat fg(nmsmat.rows, nmsmat.cols, CV_64F);
		for (int j = 0; j < nmsmat.rows; j++){
			for (int k = 0; k < nmsmat.cols; k++){
				fg.at<double>(j, k) = exp(-((double(pow((Y.at<int>(j, k) - ps[i].y), 2)) / 2 / pow(sigma, 2)) + (double(pow((X.at<int>(j, k) - ps[i].x), 2)) / 2 / pow(sigma, 2))));
			}
		}
		double mymin,mymax;
		minMaxLoc(fg, &mymin, &mymax);
		if (mymax != 0){
			fg = (pnts[i] * fg) / mymax;
			GMM = GMM + fg;
		}
		
	}
	double mymin, mymax;
	minMaxLoc(GMM, &mymin, &mymax);
	if (mymax != 0){
		GMM = GMM / mymax;
	}
	return GMM;
}

double medianMat(Mat Input, int nVals){

	// COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
	float range[] = { 0, 1 };
	const float* histRange[] = { range };
	bool uniform = true; bool accumulate = false;
	Mat hist;
	calcHist(&Input, 1, 0, Mat(), hist, 1, &nVals, histRange, uniform, accumulate);
	
	//TODO! -hist image - not a must!

	//Mat histImg = Mat::zeros(sbins*scale, hbins * 10, CV_8UC3);

	//for (int h = 0; h < hbins; h++)
	//for (int s = 0; s < sbins; s++)
	//{
	//	float binVal = hist.at<float>(h, s);
	//	int intensity = cvRound(binVal * 255 / maxVal);
	//	rectangle(histImg, Point(h*scale, s*scale),
	//		Point((h + 1)*scale - 1, (s + 1)*scale - 1),
	//		Scalar::all(intensity),
	//		CV_FILLED);
	//}
	// COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
	Mat cdf;
	hist.copyTo(cdf);
	for (int i = 1; i <= nVals - 1; i++){
		cdf.at<float>(i) += cdf.at<float>(i - 1);
	}
	cdf /= Input.total();

	// COMPUTE MEDIAN
	double medianVal;
	for (int i = 0; i <= nVals - 1; i++){
		if (cdf.at<float>(i) >= 0.95) { medianVal = i;  break; }
	}
	return medianVal / nVals;
}

// Init and start detector
void run_detect() {
	// Init forest with number of trees
	CRForest crForest( ntrees ); ////TODO for distributed system (detecting)

	// Load forest
	crForest.loadForest(treepath.c_str());	

	// Init detector
	CRForestDetector crDetect(&crForest, p_width, p_height);

	// create directory for output
	string execstr = "mkdir ";
	execstr += outpath;
	system( execstr.c_str() );

	// run detector
	detect(crDetect);
}

// Init and start training
void run_train() {
	// Init forest with number of trees
	CRForest crForest( ntrees ); 

	// Init random generator
	time_t t = time(NULL);
	int seed = (int)t;

	CvRNG cvRNG(seed);
						
	// Create directory
	string tpath(treepath);
	tpath.erase(tpath.find_last_of(PATH_SEP));
	if (!DirectoryExists(tpath.c_str())){
		string execstr = "mkdir ";
		execstr += tpath;
		system(execstr.c_str());
	}

	// Init training data
	CRPatch Train(&cvRNG, p_width, p_height, 2); 
			
	// Extract training patches
	extract_Patches(Train, &cvRNG); 

	// Train forest (minimum patches,max_depth,Trainset,# of different tests to randomly create)
	crForest.trainForest(20, 15, &cvRNG, Train, 2000);

	// Save forest
	crForest.saveForest(treepath.c_str(), off_tree);

}

void run_post(int sigmaSup = 9) {


	// Load image names
	vector<string> vFilenames;
	loadImFile(vFilenames);

	char buffer[200];

	// Storage for output
	Mat nmsmat, gmmmat;

	// Run detector for each image
	for (unsigned int i = 0; i<vFilenames.size(); ++i) {
		// Load image
		//IplImage *img = 0;
		Mat img = imread(vFilenames[i].c_str(), CV_LOAD_IMAGE_UNCHANGED); //previous implementaion save .png (8-bit) I'm saving .png (16bit uint)
		if (img.data==NULL) {
			cout << "Could not load image file: " << (impath + "/" + vFilenames[i]).c_str() << endl;
			exit(-1);
		}
		//TODO!
		// Scalar mean, std;
		// meanStdDev(cvarrToMat(img), mean, std);
		//if (mean < 255 / 4) || std < 20 
		Mat imgdbl;
		img.convertTo(imgdbl, CV_32F, 1. / out_scale);
		// Blurring the result witn a 9x9 2-sigma 
		Mat blur;
		GaussianBlur(imgdbl, blur, Size(9, 9), 2, 2);
		double med=medianMat(blur, pow(2, 16));
		Mat blurth;
		Mat BW = (blur >= med); // blur above TH - Binary map
		vector<vector<Point> > contours;
		findContours(BW, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		vector<int> idx;
		double maxarea = -1;
		int  maxidx = -1;
		if (contours.size() > 1){
			for (size_t k = 0; k < contours.size(); k++){
				double area = contourArea(contours[k]);
				if (area > maxarea){
					maxarea = area;
					maxidx = k;
				}
				if (area > 20 * 20){
					idx.push_back(k);
					//cout << "Area is: "<< area << " Contour idx: " << k << endl;
				}
			}
		}
		else // only one contour best guess is its peak
			idx.push_back(0);
		Mat BWfilt = Mat::zeros(BW.rows, BW.cols, CV_8UC1);
		if (idx.empty()) // there are lots of peaks but none is significant - best guess is the middle of the frame
			BWfilt.at<uchar>(Point(round(BW.cols / 2),round(BW.rows / 2))) = 255;
		else {
			for (size_t k = 0; k < idx.size(); k++)
				drawContours(BWfilt, contours, idx.at(k), Scalar(255), -1, 8, noArray(), 0);
		}
		blur.copyTo(blurth, BWfilt); 

		// Non-maximal suppression
		// nonMaximaSuppression(blurth, sigmaSup, nmsmat, Mat());
		//nonMaximaSuppression(cvarrToMat(img), sigmaSup, nmsmat, Mat());
		//double maxnmsmat;
		//minMaxLoc(nmsmat,NULL,&maxnmsmat);
		//if (maxnmsmat == 0) { // no maximum output from NMS pick the maimum of blurth
		//	Point pnt;
		//	minMaxLoc(blurth, NULL, NULL, NULL, &pnt);
		//	nmsmat.at<uchar>(pnt) = 1;
		//}
		// Gaussian of maximas
		gmmmat = nmsMat2GMM(BWfilt, imgdbl);
		// Store result
		string delimiter = ".";
		string s = vFilenames[i].c_str();
		string token = s.substr(0, s.find(delimiter)); //fullpath without file extention
		size_t found = token.find_last_of("/\\");
		string fname = token.substr(found + 1);//filename
		string path = token.substr(0, found); //full path of image without filename
		string pfolder = path.substr(path.find_last_of("/\\'") + 1); //parent folder only
		string curfolder = outpath + "\\" + pfolder; //store path for detection
		// Check if folder for result is exist and create if not
		if (!DirectoryExists(curfolder.c_str())){
			string execstr1 = "mkdir ";
			execstr1 += curfolder;
			system(execstr1.c_str());
		}
		sprintf_s(buffer, "%s\\%s.png", curfolder.c_str(), fname.c_str());
		Mat final;
		gmmmat = gmmmat * 255;
		gmmmat.convertTo(final, CV_8UC1);
		imwrite(buffer, final);
		//cvSaveImage( buffer, tmp );
		// Release image
		//cvReleaseImage(&img);
		cout << fname.c_str() << endl;
	}

}

int main(int argc, char* argv[])
{
	int mode = 1;

	// Check argument
	if(argc<2) {
		cout << "Usage: CRForest-Detector.exe mode [config.txt] [tree_offset]" << endl;
		cout << "mode: 0 - train; 1 - show; 2 - detect" << endl;
		cout << "tree_offset: output number for trees" << endl;
		cout << "Load default: mode - 2" << endl; 
	} else
		mode = atoi(argv[1]);
	
	off_tree = 0;	
	if(argc>3)
		off_tree = atoi(argv[3]);

	// load configuration for dataset
	if(argc>2)
		loadConfig(argv[2], mode);
	else
		loadConfig("config.txt", mode);
	int tstart_train = clock();
	switch ( mode ) { 
		case 0: 	
			// train forest
			cout << "Initiate Timer for training" << endl;
			run_train();
			cout << "Time for Training:" << (double)(clock() - tstart_train) / CLOCKS_PER_SEC << " sec" << endl;
			break;	

		case 1: 
					
			// train forest
			show();
			break;

		case 3:
		//post processing: 1. Non-maximal supression 2. Gaussian Kernel convolution
			run_post();
			break;
		

		default:

			// detection
			run_detect();
			break;
	}

	system("pause");
	return 0;
}



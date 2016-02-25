/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRPatch.h"
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <stdio.h>
//#include <stdlib.h>
#include <deque>

using namespace std;
// extract patches for training (thery are saved inside CRPatch in vLPatches[label=0/1]
void CRPatch::extractPatches(IplImage *img,const char* fullpath, unsigned int n, int label, CvRect* box, std::vector<CvPoint>* vCenter) {
	// extract features
	vector<IplImage*> vImg;
	//extractFeatureChannelsExtra(img, vImg,fullpath);
	extractPCAChannelsPlusEst(img, vImg, fullpath);
	
	CvMat tmp;
	int offx = width / 2;
	int offy = height / 2;

	// generate x,y locations
	CvMat* locations = cvCreateMat(n, 1, CV_32SC2);
	if (box == 0)
		cvRandArr(cvRNG, locations, CV_RAND_UNI, cvScalar(0, 0, 0, 0), cvScalar(img->width - width, img->height - height, 0, 0));
	else
		cvRandArr(cvRNG, locations, CV_RAND_UNI, cvScalar(box->x, box->y, 0, 0), cvScalar(box->x + box->width - width + 1, box->y + box->height - height + 1, 0, 0));

	// reserve memory
	unsigned int offset = vLPatches[label].size();
	vLPatches[label].reserve(offset + n);
	for (unsigned int i = 0; i<n; ++i) {
		CvPoint pt = *(CvPoint*)cvPtr1D(locations, i, 0);//transforming the generated locations to 1D array pt

		PatchFeature pf;
		vLPatches[label].push_back(pf);

		vLPatches[label].back().roi.x = pt.x;  vLPatches[label].back().roi.y = pt.y;
		vLPatches[label].back().roi.width = width;  vLPatches[label].back().roi.height = height;
		vLPatches[label].back().fg = label;

		if (vCenter != 0) {
			// saving the offset from the peak of distribution for all channels @ center.x/y
			vLPatches[label].back().center.resize(vCenter->size());
			for (unsigned int c = 0; c<vCenter->size(); ++c) {
				vLPatches[label].back().center[c].x = pt.x + offx - (*vCenter)[c].x;//vCenter is the center of the GT patch
				vLPatches[label].back().center[c].y = pt.y + offy - (*vCenter)[c].y;
			}
		}

		//saving all feature channels of patch
		vLPatches[label].back().vPatch.resize(vImg.size());
		for (unsigned int c = 0; c<vImg.size(); ++c) {
			cvGetSubRect(vImg[c], &tmp, vLPatches[label].back().roi);
			vLPatches[label].back().vPatch[c] = cvCloneMat(&tmp);
		}

	}

	cvReleaseMat(&locations);
	for (unsigned int c = 0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);
}

void CRPatch::extractPatchesMul(IplImage *img, const char* fullpath, unsigned int n, int label, std::vector<CvRect>& box, int startpos,int endpos) {
	// extract features
	vector<IplImage*> vImg;
	//extractFeatureChannelsExtra(img, vImg, fullpath);
	extractPCAChannelsPlusEst(img, vImg, fullpath);

	CvMat tmp;
	int offx = width / 2;
	int offy = height / 2;
	for (int j = startpos; j < endpos; j++){
		// generate x,y locations
		CvMat* locations = cvCreateMat(n, 1, CV_32SC2);
		cvRandArr(cvRNG, locations, CV_RAND_UNI, cvScalar(box[j].x, box[j].y, 0, 0), cvScalar(box[j].x + box[j].width - width + 1, box[j].y + box[j].height - height + 1, 0, 0));

		// reserve memory
		unsigned int offset = vLPatches[label].size();
		vLPatches[label].reserve(offset + n);
		for (unsigned int i = 0; i < n; ++i) {
			CvPoint pt = *(CvPoint*)cvPtr1D(locations, i, 0);//transforming the generated locations to 1D array pt

			PatchFeature pf;
			vLPatches[label].push_back(pf);

			vLPatches[label].back().roi.x = pt.x;  vLPatches[label].back().roi.y = pt.y;
			vLPatches[label].back().roi.width = width;  vLPatches[label].back().roi.height = height;
			vLPatches[label].back().fg = label;

			//saving all feature channels of patch
			vLPatches[label].back().vPatch.resize(vImg.size());
			for (unsigned int c = 0; c < vImg.size(); ++c) {
				cvGetSubRect(vImg[c], &tmp, vLPatches[label].back().roi);
				vLPatches[label].back().vPatch[c] = cvCloneMat(&tmp);
			}

		}
		cvReleaseMat(&locations);
	}
	for (unsigned int c = 0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);
}

void CRPatch::extractPatches_orig(IplImage *img, unsigned int n, int label, CvRect* box, std::vector<CvPoint>* vCenter) {
	// extract features
	vector<IplImage*> vImg;
	extractFeatureChannels(img, vImg);

	CvMat tmp;
	int offx = width/2; 
	int offy = height/2;

	// generate x,y locations
	CvMat* locations = cvCreateMat( n, 1, CV_32SC2 );
	if(box==0)
		cvRandArr( cvRNG, locations, CV_RAND_UNI, cvScalar(0,0,0,0), cvScalar(img->width-width,img->height-height,0,0) );
	else
		cvRandArr( cvRNG, locations, CV_RAND_UNI, cvScalar(box->x,box->y,0,0), cvScalar(box->x+box->width-width+1,box->y+box->height-height+1,0,0) );

	// reserve memory
	unsigned int offset = vLPatches[label].size();
	vLPatches[label].reserve(offset+n);
	for(unsigned int i=0; i<n; ++i) {
		CvPoint pt = *(CvPoint*)cvPtr1D( locations, i, 0 );
		
		PatchFeature pf;
		vLPatches[label].push_back(pf);

		vLPatches[label].back().roi.x = pt.x;  vLPatches[label].back().roi.y = pt.y;  
		vLPatches[label].back().roi.width = width;  vLPatches[label].back().roi.height = height; 

		if(vCenter!=0) {
			vLPatches[label].back().center.resize(vCenter->size());
			for(unsigned int c = 0; c<vCenter->size(); ++c) {
				vLPatches[label].back().center[c].x = pt.x + offx - (*vCenter)[c].x;
				vLPatches[label].back().center[c].y = pt.y + offy - (*vCenter)[c].y;
			}
		}

		vLPatches[label].back().vPatch.resize(vImg.size());
		for(unsigned int c=0; c<vImg.size(); ++c) {
			cvGetSubRect( vImg[c], &tmp,  vLPatches[label].back().roi );
			vLPatches[label].back().vPatch[c] = cvCloneMat(&tmp);
		}

	}

	cvReleaseMat(&locations);
	for(unsigned int c=0; c<vImg.size(); ++c)
		cvReleaseImage(&vImg[c]);
}

void CRPatch::extractFeatureChannels(IplImage *img, std::vector<IplImage*>& vImg) {
	// 32 feature channels
	// 7+9 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
	// 16+16 channels: minfilter + maxfilter on 5x5 neighborhood 

	vImg.resize(32);
	for(unsigned int c=0; c<vImg.size(); ++c)
		vImg[c] = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U , 1); 

	// Get intensity
	cvCvtColor( img, vImg[0], CV_RGB2GRAY );

	// Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
	IplImage* I_x = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1); 
	IplImage* I_y = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1); 
	
	// |I_x|, |I_y|
	cvSobel(vImg[0],I_x,1,0,3);			

	cvSobel(vImg[0],I_y,0,1,3);			

	cvConvertScaleAbs( I_x, vImg[3], 0.25);

	cvConvertScaleAbs( I_y, vImg[4], 0.25);
	
	{
	  short* dataX;
	  short* dataY;
	  uchar* dataZ;
	  int stepX, stepY, stepZ;
	  CvSize size;
	  int x, y;

	  cvGetRawData( I_x, (uchar**)&dataX, &stepX, &size);
	  cvGetRawData( I_y, (uchar**)&dataY, &stepY);
	  cvGetRawData( vImg[1], (uchar**)&dataZ, &stepZ);
	  stepX /= sizeof(dataX[0]);
	  stepY /= sizeof(dataY[0]);
	  stepZ /= sizeof(dataZ[0]);
	  
	  // Orientation of gradients
	  for( y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ  )
	    for( x = 0; x < size.width; x++ ) {
	      // Avoid division by zero
	      float tx = (float)dataX[x] + (float)_copysign(0.000001f, (float)dataX[x]);
	      // Scaling [-pi/2 pi/2] -> [0 80*pi]
	      dataZ[x]=uchar( ( atan((float)dataY[x]/tx)+3.14159265f/2.0f ) * 80 ); 
	    }
	}
	
	{
	  short* dataX;
	  short* dataY;
	  uchar* dataZ;
	  int stepX, stepY, stepZ;
	  CvSize size;
	  int x, y;
	  
	  cvGetRawData( I_x, (uchar**)&dataX, &stepX, &size);
	  cvGetRawData( I_y, (uchar**)&dataY, &stepY);
	  cvGetRawData( vImg[2], (uchar**)&dataZ, &stepZ);
	  stepX /= sizeof(dataX[0]);
	  stepY /= sizeof(dataY[0]);
	  stepZ /= sizeof(dataZ[0]);
	  
	  // Magnitude of gradients
	  for( y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ  )
	    for( x = 0; x < size.width; x++ ) {
	      dataZ[x] = (uchar)( sqrt((float)dataX[x]*(float)dataX[x] + (float)dataY[x]*(float)dataY[x]) );
	    }
	}

	// 9-bin HOG feature stored at vImg[7] - vImg[15] 
	hog.extractOBin(vImg[1], vImg[2], vImg, 7);
	
	// |I_xx|, |I_yy|

	cvSobel(vImg[0],I_x,2,0,3);
	cvConvertScaleAbs( I_x, vImg[5], 0.25);	
	
	cvSobel(vImg[0],I_y,0,2,3);
	cvConvertScaleAbs( I_y, vImg[6], 0.25);
	
	// L, a, b
	cvCvtColor( img, img, CV_RGB2Lab  );

	cvReleaseImage(&I_x);
	cvReleaseImage(&I_y);	
	
	cvSplit( img, vImg[0], vImg[1], vImg[2], 0);

	// min filter
	for(int c=0; c<16; ++c)
		minfilt(vImg[c], vImg[c+16], 5);

	//max filter
	for(int c=0; c<16; ++c)
		maxfilt(vImg[c], 5);


	
#if 0
	// for debugging only
	char buffer[40];
	for(unsigned int i = 0; i<vImg.size();++i) {
		sprintf_s(buffer,"out-%d.png",i);
		cvNamedWindow(buffer,1);
		cvShowImage(buffer, vImg[i]);
		//cvSaveImage( buffer, vImg[i] );
	}

	cvWaitKey();

	for(unsigned int i = 0; i<vImg.size();++i) {
		sprintf_s(buffer,"%d",i);
		cvDestroyWindow(buffer);
	}
#endif


}

void CRPatch::extractFeatureChannelsExtra(IplImage *img, std::vector<IplImage*>& vImg, const char* fullpath) {
	// 36 feature channels
	// 7+9 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
	// 2 channels : PCAm PCAs
	// 18+18 channels: minfilter + maxfilter on 5x5 neighborhood 

	vImg.resize(36);
	for (unsigned int c = 0; c<vImg.size(); ++c)
		vImg[c] = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);

	// Get intensity
	cvCvtColor(img, vImg[0], CV_RGB2GRAY);

	// Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
	IplImage* I_x = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_16S, 1);
	IplImage* I_y = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_16S, 1);

	// |I_x|, |I_y|
	cvSobel(vImg[0], I_x, 1, 0, 3);

	cvSobel(vImg[0], I_y, 0, 1, 3);

	cvConvertScaleAbs(I_x, vImg[3], 0.25);

	cvConvertScaleAbs(I_y, vImg[4], 0.25);

	{
		short* dataX;
		short* dataY;
		uchar* dataZ;
		int stepX, stepY, stepZ;
		CvSize size;
		int x, y;

		cvGetRawData(I_x, (uchar**)&dataX, &stepX, &size);
		cvGetRawData(I_y, (uchar**)&dataY, &stepY);
		cvGetRawData(vImg[1], (uchar**)&dataZ, &stepZ);
		stepX /= sizeof(dataX[0]);
		stepY /= sizeof(dataY[0]);
		stepZ /= sizeof(dataZ[0]);

		// Orientation of gradients
		for (y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ)
		for (x = 0; x < size.width; x++) {
			// Avoid division by zero
			float tx = (float)dataX[x] + (float)_copysign(0.000001f, (float)dataX[x]);
			// Scaling [-pi/2 pi/2] -> [0 80*pi]
			dataZ[x] = uchar((atan((float)dataY[x] / tx) + 3.14159265f / 2.0f) * 80);
		}
	}

	{
		short* dataX;
		short* dataY;
		uchar* dataZ;
		int stepX, stepY, stepZ;
		CvSize size;
		int x, y;

		cvGetRawData(I_x, (uchar**)&dataX, &stepX, &size);
		cvGetRawData(I_y, (uchar**)&dataY, &stepY);
		cvGetRawData(vImg[2], (uchar**)&dataZ, &stepZ);
		stepX /= sizeof(dataX[0]);
		stepY /= sizeof(dataY[0]);
		stepZ /= sizeof(dataZ[0]);

		// Magnitude of gradients
		for (y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ)
		for (x = 0; x < size.width; x++) {
			dataZ[x] = (uchar)(sqrt((float)dataX[x] * (float)dataX[x] + (float)dataY[x] * (float)dataY[x]));
		}
	}

	// 9-bin HOG feature stored at vImg[7] - vImg[15] 
	hog.extractOBin(vImg[1], vImg[2], vImg, 7);

	// |I_xx|, |I_yy|

	cvSobel(vImg[0], I_x, 2, 0, 3);
	cvConvertScaleAbs(I_x, vImg[5], 0.25);

	cvSobel(vImg[0], I_y, 0, 2, 3);
	cvConvertScaleAbs(I_y, vImg[6], 0.25);

	// L, a, b
	cvCvtColor(img, img, CV_RGB2Lab);

	cvReleaseImage(&I_x);
	cvReleaseImage(&I_y);

	cvSplit(img, vImg[0], vImg[1], vImg[2], 0);

	// Get PCAm and PCAs Channel from saved images
	string delimiter = ".";
	string s = fullpath;
	string token = s.substr(0, s.find(delimiter)); //fullpath without file extention
	size_t found = token.find_last_of("/\\");
	string fname = token.substr(found + 1);//filename
	string path = token.substr(0, found); //full path of image without filename
	string pfolder = path.substr(path.find_last_of("/\\'") + 1); //parent folder only
	string rootpath = path.substr(0, path.find_last_of("/\\'"));
	rootpath = rootpath.substr(0, rootpath.find_last_of("/\\'"));
	string fullpathPCAm = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAm.png";
	string fullpathPCAs = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAs.png";

	vImg[16] = cvLoadImage(fullpathPCAm.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	vImg[17] = cvLoadImage(fullpathPCAs.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	// min filter
	for (int c = 0; c<18; ++c)
		minfilt(vImg[c], vImg[c + 18], 5);

	//max filter
	for (int c = 0; c<18; ++c)
		maxfilt(vImg[c], 5);



#if 0
	// for debugging only
	char buffer[40];
	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "out-%d.png", i);
		cvNamedWindow(buffer, 1);
		cvShowImage(buffer, vImg[i]);
		//cvSaveImage( buffer, vImg[i] );
	}

	cvWaitKey();

	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "%d", i);
		cvDestroyWindow(buffer);
	}
#endif


}

void CRPatch::extractPCAChannels(IplImage *img, std::vector<IplImage*>& vImg, const char* fullpath) {
	// 4 feature channels
	// 0 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
	// 2 channels : PCAm PCAs
	// 2+2 channels: minfilter + maxfilter on 5x5 neighborhood 

	vImg.resize(4);
	for (unsigned int c = 0; c<vImg.size(); ++c)
		vImg[c] = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);

	// Get PCAm and PCAs Channel from saved images
	string delimiter = ".";
	string s = fullpath;
	string token = s.substr(0, s.find(delimiter)); //fullpath without file extention
	size_t found = token.find_last_of("/\\");
	string fname = token.substr(found + 1);//filename
	string path = token.substr(0, found); //full path of image without filename
	string pfolder = path.substr(path.find_last_of("/\\'") + 1); //parent folder only
	string rootpath = path.substr(0, path.find_last_of("/\\'"));
	rootpath = rootpath.substr(0, rootpath.find_last_of("/\\'"));
	string fullpathPCAm = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAm.png";
	string fullpathPCAs = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAs.png";

	vImg[0] = cvLoadImage(fullpathPCAm.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	vImg[1] = cvLoadImage(fullpathPCAs.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	// min filter
	for (int c = 0; c<2; ++c)
		minfilt(vImg[c], vImg[c + 2], 5);

	//max filter
	for (int c = 0; c<2; ++c)
		maxfilt(vImg[c], 5);



#if 0
	// for debugging only
	char buffer[40];
	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "out-%d.png", i);
		cvNamedWindow(buffer, 1);
		cvShowImage(buffer, vImg[i]);
		//cvSaveImage( buffer, vImg[i] );
	}

	cvWaitKey();

	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "%d", i);
		cvDestroyWindow(buffer);
	}
#endif


}

void CRPatch::extractPCAChannelsPlusEst(IplImage *img, std::vector<IplImage*>& vImg, const char* fullpath) {
	// 6 feature channels
	// 0 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
	// 2 channels : PCAm PCAs
	// 1 channel : GT(t-1) || Gaussian in middle frame in training and Estimation of frame(t-1) || Gaussian in middle frame in testing
	// 2+1+2+1 channels: minfilter + maxfilter on 5x5 neighborhood 

	vImg.resize(6);
	for (unsigned int c = 0; c<vImg.size(); ++c)
		vImg[c] = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);

	// Get PCAm and PCAs Channel from saved images
	string delimiter = ".";
	string s = fullpath;
	string token = s.substr(0, s.find(delimiter)); //fullpath without file extention
	size_t found = token.find_last_of("/\\");
	string fname = token.substr(found + 1);//filename
	string path = token.substr(0, found); //full path of image without filename
	string pfolder = path.substr(path.find_last_of("/\\'") + 1); //parent folder only
	string rootpath = path.substr(0, path.find_last_of("/\\'"));
	rootpath = rootpath.substr(0, rootpath.find_last_of("/\\'"));
	string fullpathPCAm = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAm.png";
	string fullpathPCAs = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAs.png";
	// PCA channels
	vImg[0] = cvLoadImage(fullpathPCAm.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	vImg[1] = cvLoadImage(fullpathPCAs.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	// GT channel
	std::string::size_type sz;   // alias of size_t
	int i_dec = std::stoi(fname, &sz);
	char buffer[7];

	if(i_dec > 1) // frames are starting at number 1 and format is  %06i
	{
		sprintf(buffer, "%06d", i_dec - 1);
		string fullpathGT = rootpath + "/\\" + "DIEMFIXATIONpng/\\" + pfolder + "/\\" + buffer + "_predMap.png";
		vImg[2] = cvLoadImage(fullpathGT.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	}
	else // 1st file no ground truth of previous need to put Gaussian in the middle
	{
		cv::Mat fixmat(img->height, img->width, CV_64F, cv::Scalar::all(0));
		if (img->width % 2 && img->height % 2)
		{
			fixmat.at<double>(ceil(img->width / 2),ceil( img->height / 2)) = 1;
		}
		else
		{
			cv::Rect r(img->width / 2 -1, img->height / 2-1, 2, 2);
			cv::Mat roi(fixmat, r);
			roi = 1;
		}
		vImg[2] = new IplImage(fixMat2GMM(fixmat));
	}
	// min filter
	for (int c = 0; c<3; ++c)
		minfilt(vImg[c], vImg[c + 3], 5);

	//max filter
	for (int c = 0; c<3; ++c)
		maxfilt(vImg[c], 5);



#if 0
	// for debugging only
	char buffer[40];
	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "out-%d.png", i);
		cvNamedWindow(buffer, 1);
		cvShowImage(buffer, vImg[i]);
		//cvSaveImage( buffer, vImg[i] );
	}

	cvWaitKey();

	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "%d", i);
		cvDestroyWindow(buffer);
	}
#endif


}

void CRPatch::extractPCAChannelsPlusEstTest(IplImage *img, std::vector<IplImage*>& vImg, const char* fullpath, const char* exp_fold) {
	// 6 feature channels
	// 0 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
	// 2 channels : PCAm PCAs
	// 1 channel : Estimation(t-1) || Gaussian in middle frame in training and Estimation of frame(t-1) || Gaussian in middle frame in testing
	// 2+1+2+1 channels: minfilter + maxfilter on 5x5 neighborhood 

	vImg.resize(6);
	for (unsigned int c = 0; c<vImg.size(); ++c)
		vImg[c] = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);

	// Get PCAm and PCAs Channel from saved images
	string delimiter = ".";
	string s = fullpath;
	string token = s.substr(0, s.find(delimiter)); //fullpath without file extention
	size_t found = token.find_last_of("/\\");
	string fname = token.substr(found + 1);//filename
	string path = token.substr(0, found); //full path of image without filename
	string pfolder = path.substr(path.find_last_of("/\\'") + 1); //parent folder only
	string rootpath = path.substr(0, path.find_last_of("/\\'"));
	rootpath = rootpath.substr(0, rootpath.find_last_of("/\\'"));
	string fullpathPCAm = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAm.png";
	string fullpathPCAs = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAs.png";
	// PCA channels
	vImg[0] = cvLoadImage(fullpathPCAm.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	vImg[1] = cvLoadImage(fullpathPCAs.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	// GT channel
	std::string::size_type sz;   // alias of size_t
	int i_dec = std::stoi(fname, &sz);
	char buffer[7];

	if (i_dec > 1) // frames are starting at number 1 and format is  %06i
	{
		sprintf(buffer, "%06d", i_dec - 1);
		// find previous prediction and use it.
		string exp = exp_fold;
		string fullpathGT = exp + "/\\" + buffer + "_sc0_c0_predmap.png";
		vImg[2] = cvLoadImage(fullpathGT.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	}
	else // 1st file no ground truth of previous need to put Gaussian in the middle
	{
		cv::Mat fixmat(img->height, img->width, CV_64F, cv::Scalar::all(0));
		if (img->width % 2 && img->height % 2)
		{
			fixmat.at<double>(ceil(img->width / 2), ceil(img->height / 2)) = 1;
		}
		else
		{
			cv::Rect r(img->width / 2 - 1, img->height / 2 - 1, 2, 2);
			cv::Mat roi(fixmat, r);
			roi = 1;
		}
		cv::Mat tmp = fixMat2GMM(fixmat);
		cv::Mat tmp1;
		tmp.convertTo(tmp1, CV_8U,255);

		vImg[2] = cvCloneImage(&(IplImage)tmp);
	}
	// min filter
	for (int c = 0; c<3; ++c)
		minfilt(vImg[c], vImg[c + 3], 5);

	//max filter
	for (int c = 0; c<3; ++c)
		maxfilt(vImg[c], 5);



#if 0
	// for debugging only
	char buffer[40];
	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "out-%d.png", i);
		cvNamedWindow(buffer, 1);
		cvShowImage(buffer, vImg[i]);
		//cvSaveImage( buffer, vImg[i] );
	}

	cvWaitKey();

	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "%d", i);
		cvDestroyWindow(buffer);
	}
#endif


}

void CRPatch::extractFeatureChannelsPartial(IplImage *img, std::vector<IplImage*>& vImg, const char* fullpath ) {
	// 18 feature channels + 4 features (PCAs , PCAm)x2
	// 9 channels: HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
	// 9+9 channels: minfilter + maxfilter on 5x5 neighborhood 
	//    L   ,   a   ,   b   ,| I_x |,| I_y |, | I_xx | ,| I_yy |
	// vImg[0],vImg[1],vImg[2],vImg[3],vImg[4],  vImg[5] ,vImg[6]
	
	vImg.resize(22);
	// Initialize
	for (unsigned int c = 0; c<vImg.size(); ++c)
		vImg[c] = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);

	// Get intensity
	cvCvtColor(img, vImg[0], CV_RGB2GRAY);

	// Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
	IplImage* I_x = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_16S, 1);
	IplImage* I_y = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_16S, 1);

	// |I_x|, |I_y| Gradients Magnitudes
	cvSobel(vImg[0], I_x, 1, 0, 3);
	cvSobel(vImg[0], I_y, 0, 1, 3);

	{
		short* dataX;
		short* dataY;
		uchar* dataZ;
		int stepX, stepY, stepZ;
		CvSize size;
		int x, y;

		cvGetRawData(I_x, (uchar**)&dataX, &stepX, &size);
		cvGetRawData(I_y, (uchar**)&dataY, &stepY);
		cvGetRawData(vImg[0], (uchar**)&dataZ, &stepZ);
		stepX /= sizeof(dataX[0]);
		stepY /= sizeof(dataY[0]);
		stepZ /= sizeof(dataZ[0]);

		// Orientation of gradients
		for (y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ)
		for (x = 0; x < size.width; x++) {
			// Avoid division by zero
			float tx = (float)dataX[x] + (float)_copysign(0.000001f, (float)dataX[x]);
			// Scaling [-pi/2 pi/2] -> [0 80*pi]
			dataZ[x] = uchar((atan((float)dataY[x] / tx) + 3.14159265f / 2.0f) * 80);
		}
	}

	{
		short* dataX;
		short* dataY;
		uchar* dataZ;
		int stepX, stepY, stepZ;
		CvSize size;
		int x, y;

		cvGetRawData(I_x, (uchar**)&dataX, &stepX, &size);
		cvGetRawData(I_y, (uchar**)&dataY, &stepY);
		cvGetRawData(vImg[1], (uchar**)&dataZ, &stepZ);
		stepX /= sizeof(dataX[0]);
		stepY /= sizeof(dataY[0]);
		stepZ /= sizeof(dataZ[0]);

		// Magnitude of gradients
		for (y = 0; y < size.height; y++, dataX += stepX, dataY += stepY, dataZ += stepZ)
		for (x = 0; x < size.width; x++) {
			dataZ[x] = (uchar)(sqrt((float)dataX[x] * (float)dataX[x] + (float)dataY[x] * (float)dataY[x]));
		}
	}

	// 9-bin HOG feature stored at vImg[2] - vImg[11] 
	hog.extractOBin(vImg[0], vImg[1], vImg, 2);
	
	// Get PCAm and PCAs Channel from saved images
	string delimiter = ".";
	string s = fullpath;
	string token = s.substr(0, s.find(delimiter)); //fullpath without file extention
	size_t found = token.find_last_of("/\\");
	string fname = token.substr(found + 1);//filename
	string path = token.substr(0, found); //full path of image without filename
	string pfolder = path.substr(path.find_last_of("/\\'")+1); //parent folder only
	string rootpath = path.substr(0, path.find_last_of("/\\'"));
	rootpath = rootpath.substr(0, rootpath.find_last_of("/\\'"));
	string fullpathPCAm = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAm.png";
	string fullpathPCAs = rootpath + "/\\" + "DIEMPCApng/\\" + pfolder + "/\\" + fname + "_PCAs.png";

	vImg[0] = cvLoadImage(fullpathPCAm.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	vImg[1] = cvLoadImage(fullpathPCAs.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	// min filter
	for (int c = 0; c<11; ++c)
		minfilt(vImg[c], vImg[c + 11], 5);

	//max filter
	for (int c = 0; c<11; ++c)
		maxfilt(vImg[c], 5);
#if 0
	// for debugging only
	char buffer[40];
	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "out-%d.png", i);
		cvNamedWindow(buffer, 1);
		cvShowImage(buffer, vImg[i]);
		//cvSaveImage( buffer, vImg[i] );
	}

	cvWaitKey();

	for (unsigned int i = 0; i<vImg.size(); ++i) {
		sprintf_s(buffer, "%d", i);
		cvDestroyWindow(buffer);
	}
#endif
}

void CRPatch::maxfilt(IplImage *src, unsigned int width) {

	uchar* s_data;
	int step;
	CvSize size;

	cvGetRawData( src, (uchar**)&s_data, &step, &size );
	step /= sizeof(s_data[0]);

	for(int  y = 0; y < size.height; y++) {
		maxfilt(s_data+y*step, 1, size.width, width);
	}

	cvGetRawData( src, (uchar**)&s_data);

	for(int  x = 0; x < size.width; x++)
		maxfilt(s_data+x, step, size.height, width);

}

void CRPatch::maxfilt(IplImage *src, IplImage *dst, unsigned int width) {

	uchar* s_data;
	uchar* d_data;
	int step;
	CvSize size;

	cvGetRawData( src, (uchar**)&s_data, &step, &size );
	cvGetRawData( dst, (uchar**)&d_data, &step, &size );
	step /= sizeof(s_data[0]);

	for(int  y = 0; y < size.height; y++)
		maxfilt(s_data+y*step, d_data+y*step, 1, size.width, width);

	cvGetRawData( src, (uchar**)&d_data);

	for(int  x = 0; x < size.width; x++)
		maxfilt(d_data+x, step, size.height, width);

}

void CRPatch::minfilt(IplImage *src, unsigned int width) {

	uchar* s_data;
	int step;
	CvSize size;

	cvGetRawData( src, (uchar**)&s_data, &step, &size );
	step /= sizeof(s_data[0]);

	for(int  y = 0; y < size.height; y++)
		minfilt(s_data+y*step, 1, size.width, width);

	cvGetRawData( src, (uchar**)&s_data);

	for(int  x = 0; x < size.width; x++)
		minfilt(s_data+x, step, size.height, width);

}

void CRPatch::minfilt(IplImage *src, IplImage *dst, unsigned int width) {

	uchar* s_data;
	uchar* d_data;
	int step;
	CvSize size;

	cvGetRawData( src, (uchar**)&s_data, &step, &size );
	cvGetRawData( dst, (uchar**)&d_data, &step, &size );
	step /= sizeof(s_data[0]);

	for(int  y = 0; y < size.height; y++)
		minfilt(s_data+y*step, d_data+y*step, 1, size.width, width);

	cvGetRawData( src, (uchar**)&d_data);

	for(int  x = 0; x < size.width; x++)
		minfilt(d_data+x, step, size.height, width);

}

void CRPatch::maxfilt(uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	maxvalues[0] = data[0];
	for(unsigned int i=0; i < d-step; i+=step) {
		for(unsigned int k=i; k<d+i; k+=step) {
			if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
		}
		maxvalues[i+step] = maxvalues[i];
	}

	maxvalues[size-step] = data[size-step];
	for(unsigned int i=size-step; i > size-d; i-=step) {
		for(unsigned int k=i; k>i-d; k-=step) {
			if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
		}
		maxvalues[i-step] = maxvalues[i];
	}

    deque<int> maxfifo;
    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			maxvalues[i-d] = data[maxfifo.size()>0 ? maxfifo.front(): i-step];
		}
    
		if(data[i] < data[i-step]) { 

			maxfifo.push_back(i-step);
			if(i==  width+maxfifo.front()) 
				maxfifo.pop_front();

		} else {

			while(maxfifo.size() > 0) {
				if(data[i] <= data[maxfifo.back()]) {
					if(i==  width+maxfifo.front()) 
						maxfifo.pop_front();
				break;
				}
				maxfifo.pop_back();
			}

		}

    }  

    maxvalues[size-d] = data[maxfifo.size()>0 ? maxfifo.front():size-step];
 
}

void CRPatch::maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	deque<uchar> tmp;

	tmp.push_back(data[0]);
	for(unsigned int k=step; k<d; k+=step) {
		if(data[k]>tmp.back()) tmp.back() = data[k];
	}

	for(unsigned int i=step; i < d-step; i+=step) {
		tmp.push_back(tmp.back());
		if(data[i+d-step]>tmp.back()) tmp.back() = data[i+d-step];
	}


    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
			data[i-width] = tmp.front();
			tmp.pop_front();
		}
    
		if(data[i] < data[i-step]) { 

			minfifo.push_back(i-step);
			if(i==  width+minfifo.front()) 
				minfifo.pop_front();

		} else {

			while(minfifo.size() > 0) {
				if(data[i] <= data[minfifo.back()]) {
					if(i==  width+minfifo.front()) 
						minfifo.pop_front();
				break;
				}
				minfifo.pop_back();
			}

		}

    }  

	tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);
	
	for(unsigned int k=size-step-step; k>=size-d; k-=step) {
		if(data[k]>data[size-step]) data[size-step] = data[k];
	}

	for(unsigned int i=size-step-step; i >= size-d; i-=step) {
		data[i] = data[i+step];
		if(data[i-d+step]>data[i]) data[i] = data[i-d+step];
	}

	for(unsigned int i=size-width; i<=size-d; i+=step) {
		data[i] = tmp.front();
		tmp.pop_front();
	}
 
}

void CRPatch::minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	minvalues[0] = data[0];
	for(unsigned int i=0; i < d-step; i+=step) {
		for(unsigned int k=i; k<d+i; k+=step) {
			if(data[k]<minvalues[i]) minvalues[i] = data[k];
		}
		minvalues[i+step] = minvalues[i];
	}

	minvalues[size-step] = data[size-step];
	for(unsigned int i=size-step; i > size-d; i-=step) {
		for(unsigned int k=i; k>i-d; k-=step) {
			if(data[k]<minvalues[i]) minvalues[i] = data[k];
		}
		minvalues[i-step] = minvalues[i];
	}

    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			minvalues[i-d] = data[minfifo.size()>0 ? minfifo.front(): i-step];
		}
    
		if(data[i] > data[i-step]) { 

			minfifo.push_back(i-step);
			if(i==  width+minfifo.front()) 
				minfifo.pop_front();

		} else {

			while(minfifo.size() > 0) {
				if(data[i] >= data[minfifo.back()]) {
					if(i==  width+minfifo.front()) 
						minfifo.pop_front();
				break;
				}
				minfifo.pop_back();
			}

		}

    }  

    minvalues[size-d] = data[minfifo.size()>0 ? minfifo.front():size-step];
 
}

void CRPatch::minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	deque<uchar> tmp;

	tmp.push_back(data[0]);
	for(unsigned int k=step; k<d; k+=step) {
		if(data[k]<tmp.back()) tmp.back() = data[k];
	}

	for(unsigned int i=step; i < d-step; i+=step) {
		tmp.push_back(tmp.back());
		if(data[i+d-step]<tmp.back()) tmp.back() = data[i+d-step];
	}


    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
			data[i-width] = tmp.front();
			tmp.pop_front();
		}
    
		if(data[i] > data[i-step]) { 

			minfifo.push_back(i-step);
			if(i==  width+minfifo.front()) 
				minfifo.pop_front();

		} else {

			while(minfifo.size() > 0) {
				if(data[i] >= data[minfifo.back()]) {
					if(i==  width+minfifo.front()) 
						minfifo.pop_front();
				break;
				}
				minfifo.pop_back();
			}

		}

    }  

	tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);
	
	for(unsigned int k=size-step-step; k>=size-d; k-=step) {
		if(data[k]<data[size-step]) data[size-step] = data[k];
	}

	for(unsigned int i=size-step-step; i >= size-d; i-=step) {
		data[i] = data[i+step];
		if(data[i-d+step]<data[i]) data[i] = data[i-d+step];
	}
 
	for(unsigned int i=size-width; i<=size-d; i+=step) {
		data[i] = tmp.front();
		tmp.pop_front();
	}
}

void CRPatch::maxminfilt(uchar* data, uchar* maxvalues, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width) {

	unsigned int d = int((width+1)/2)*step; 
	size *= step;
	width *= step;

	maxvalues[0] = data[0];
	minvalues[0] = data[0];
	for(unsigned int i=0; i < d-step; i+=step) {
		for(unsigned int k=i; k<d+i; k+=step) {
			if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
			if(data[k]<minvalues[i]) minvalues[i] = data[k];
		}
		maxvalues[i+step] = maxvalues[i];
		minvalues[i+step] = minvalues[i];
	}

	maxvalues[size-step] = data[size-step];
	minvalues[size-step] = data[size-step];
	for(unsigned int i=size-step; i > size-d; i-=step) {
		for(unsigned int k=i; k>i-d; k-=step) {
			if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
			if(data[k]<minvalues[i]) minvalues[i] = data[k];
		}
		maxvalues[i-step] = maxvalues[i];
		minvalues[i-step] = minvalues[i];
	}

    deque<int> maxfifo, minfifo;

    for(unsigned int i = step; i < size; i+=step) {
		if(i >= width) {
			maxvalues[i-d] = data[maxfifo.size()>0 ? maxfifo.front(): i-step];
			minvalues[i-d] = data[minfifo.size()>0 ? minfifo.front(): i-step];
		}
    
		if(data[i] > data[i-step]) { 

			minfifo.push_back(i-step);
			if(i==  width+minfifo.front()) 
				minfifo.pop_front();
			while(maxfifo.size() > 0) {
				if(data[i] <= data[maxfifo.back()]) {
					if (i==  width+maxfifo.front()) 
						maxfifo.pop_front();
					break;
				}
				maxfifo.pop_back();
			}

		} else {

			maxfifo.push_back(i-step);
			if (i==  width+maxfifo.front()) 
				maxfifo.pop_front();
			while(minfifo.size() > 0) {
				if(data[i] >= data[minfifo.back()]) {
					if(i==  width+minfifo.front()) 
						minfifo.pop_front();
				break;
				}
				minfifo.pop_back();
			}

		}

    }  

    maxvalues[size-d] = data[maxfifo.size()>0 ? maxfifo.front():size-step];
	minvalues[size-d] = data[minfifo.size()>0 ? minfifo.front():size-step];
 
}

// helper function (maybe that goes somehow easier)
void CRPatch::meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat1i &X, cv::Mat1i &Y)
{
	repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

void CRPatch::meshgridTest(const cv::Range &xgv, const cv::Range &ygv, cv::Mat1i &X, cv::Mat1i &Y)
{
	vector<int> t_x, t_y;
	for (int i = xgv.start; i < xgv.end; i++) t_x.push_back(i);
	for (int i = ygv.start; i < ygv.end; i++) t_y.push_back(i);
	meshgrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
}

cv::Mat CRPatch::fixMat2GMM(cv::Mat& nmsmat, int sigma){
	cv::Mat locations;   // output, locations of non-zero pixels 
	findNonZero(nmsmat>0, locations);
	vector<int> pnts;
	vector<CvPoint> ps;
	for (int i = 0; i < locations.rows; i++){
		ps.push_back(locations.at<cv::Point>(i));
		pnts.push_back(nmsmat.at<double>(ps[i].y, ps[i].x));
	}
	cv::Mat1i X, Y;
	meshgridTest(cv::Range(0, nmsmat.cols), cv::Range(0, nmsmat.rows), X, Y);
	cv::Mat GMM(nmsmat.rows, nmsmat.cols, CV_32F, cv::Scalar::all(0));
	for (int i = 0; i < ps.size(); i++){
		cv::Mat fg(nmsmat.rows, nmsmat.cols, CV_32F);
		for (int j = 0; j < nmsmat.rows; j++){
			for (int k = 0; k < nmsmat.cols; k++){
				fg.at<float>(j, k) = exp(-((float(pow((Y.at<int>(j, k) - ps[i].y), 2)) / 2 / pow(sigma, 2)) + (float(pow((X.at<int>(j, k) - ps[i].x), 2)) / 2 / pow(sigma, 2))));
			}
		}
		double mymin, mymax;
		cv::minMaxLoc(fg, &mymin, &mymax);
		if (mymax != 0){
			fg = (pnts[i] * fg) / mymax;
			GMM = GMM + fg;
		}

	}
	double mymin, mymax;
	cv::minMaxLoc(GMM, &mymin, &mymax);
	if (mymax != 0){
		GMM = GMM / mymax;
	}
	return GMM;
}
//
// Created by ahmed on 25.12.16.
//


#include "ParticleFilter.h"
#include <string>
#include <sstream>
#include <iostream>

using namespace std;

const int NUM_IMAGES=33;
string image_prefix = "images/";
string image_suffix = ".png";
cv::Rect bb_frame1(448,191, 38,33);

string fixedLenString(int i,int len, string prefix, string suffix){
    stringstream ss;
    ss << setw(len) << setfill('0') << i;
    string s = ss.str();
    return prefix+s+suffix;
}
int main()
{
    ParticleFilter pf;

    for(int i=1; i<NUM_IMAGES; i++){
        string text;
        string fname=fixedLenString(i,2,image_prefix,image_suffix);
        cv::Mat img=cv::imread(fname.c_str());
        cv::imshow("frame",img);
        cv::waitKey(100);
        if(i==1) pf.init(img,bb_frame1);
        else pf.track(img);
        pf.showParticles(img);
    }
    return 0;
}
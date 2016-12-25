//
// Created by ahmed on 25.12.16.
//

#include "Particle.h"

void calculateRGBhistogram(cv::Mat& img,cv::Mat& hist){
    float range[] = {0,256};
    cv::Mat gray_img;
    cv::cvtColor(img,gray_img,CV_BGR2GRAY);
    bool uniform = true; bool accumulate = false;
    int histSize = 256;
    const float* histRange = { range };
    calcHist( &gray_img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
}

Particle::Particle(cv::Mat &img, const cv::Rect &bb, cv::Mat &refhist, const cv::Point2i off) {
    cv::Rect newbb(bb);
    newbb.x += off.x;
    if(newbb.x >= img.cols || newbb.x < 0)
        newbb.x = img.cols - bb.x;
    newbb.y += off.y;
    if(newbb.y >= img.rows || newbb.y < 0)
        newbb.y = img.rows - bb.y;
    newbb.x %= img.cols;
    newbb.y %= img.rows;
    this->updateParticle(img,newbb);
    this->measureFitness(refhist);
}
void Particle::updateParticle(cv::Mat& img, cv::Rect newbb){
    this->bb = newbb;
    cv::Mat roi = img(this->bb);
    calculateRGBhistogram(roi,this->hist);
}
void Particle::measureFitness(const cv::Mat &refhist) {
    this->fitness = 1 - cv::compareHist(this->hist,refhist,CV_COMP_BHATTACHARYYA);
}

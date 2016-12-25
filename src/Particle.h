//
// Created by ahmed on 25.12.16.
//

#ifndef PARTICLE_H
#define PARTICLE_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <math.h>
#include <iomanip>

void calculateRGBhistogram(cv::Mat& img,cv::Mat& hist);
class Particle{
public:
    Particle(){fitness=0;};
    Particle(cv::Mat& img,const cv::Rect& bb, cv::Mat& refhist, const cv::Point2i off);  // present frame, sampled bb from previous frame, refhist and offset from motion model
    ~Particle(){};
    cv::Rect bb;
    cv::Mat hist;
    float fitness;
    void updateParticle(cv::Mat& img, cv::Rect newbb);
    void measureFitness(const cv::Mat& refhist);    // evaluates fitness for the present particle given refhist
};

#endif //PARTICLE_H

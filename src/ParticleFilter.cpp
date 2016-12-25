//
// Created by ahmed on 25.12.16.
//

#include "ParticleFilter.h"

float calLikelihood(float fitness,float sigma){
    return exp(- fitness/ (sigma*sigma));
}
double sampleFromGaussian(double mean,double stdev) {
    double result = 0.0;
    result = mean;
    double sum=0;
    for(int i=0;i<12;++i)
    {
        double r = (double)rand() / RAND_MAX;
        sum += -stdev + r*2*stdev;
    }
    sum/=2;
    result += sum;
    return result;
}


void ParticleFilter::init(cv::Mat &img, cv::Rect &bb) {
    for(int i=0;i<numptl;++i){
        cv::Rect randomRect = this->applyMotionModel(bb);
        if(randomRect.x + randomRect.width >= img.cols || randomRect.y+randomRect.height >= img.rows){
            randomRect.x = rng.uniform(0,img.cols - randomRect.width);
            randomRect.y = rng.uniform(0,img.rows - randomRect.height);
        }
        particles[i].updateParticle(img,randomRect);
    }
    cv::Mat roi = img(bb);
    calculateRGBhistogram(roi,this->refhist);
}
void ParticleFilter::showParticles(cv::Mat& img){
    for(int pid=0; pid<numptl; ++pid){
        cv::rectangle(img,particles[pid].bb,cv::Scalar(0,0,255*particles[pid].fitness),1);
    }
    cv::imshow("tracked",img);
    cv::waitKey(100);
}
void ParticleFilter::showParticle(cv::Mat& img,int pid)
{
    if(pid == -1) return;

    cv::Mat show=img.clone();
    cv::rectangle(show,particles[pid].bb,cv::Scalar(0,0,255*particles[pid].fitness),1);
    cv::imshow("Fittest Particle",show);
    cv::waitKey(100);
}
void ParticleFilter::normalizeWeights()
{
    double normalizer = 0;
    for(int i=0;i<particles.size();++i)
        normalizer += particles[i].fitness;
    for(int i=0;i<particles.size();++i)
        particles[i].fitness /= normalizer;
}
void ParticleFilter::evaluateCumulFeat() {
    this->cumulFit[0] = this->particles[0].fitness;
    for(int i=1;i<this->numptl;++i)
        this->cumulFit[i] = this->cumulFit[i-1] + this->particles[i].fitness;
    for(int i=1;i<this->numptl;++i)
        this->cumulFit[i] /= this->cumulFit[this->numptl-1];
}
std::vector<Particle> ParticleFilter::resample() {
    std::vector<Particle> newParticles;
    double r=((double)rand()/(double)RAND_MAX);
    r /= particles.size();
    double cummSum = particles[0].fitness;
    int idx =0;
    for(int i=0;i<particles.size();++i)
    {
        double pointer = r + (double)i/particles.size();
        while(pointer > cummSum)
        {
            ++idx;
            cummSum += particles[idx].fitness;
        }
        newParticles.push_back(particles[idx]);
    }
    return newParticles;
}

int ParticleFilter::sampleParticle() {
    float value = this->rng.uniform(0,1);
    for(int i=0;i<this->numptl;++i){
        if(value < this->cumulFit[i])
            return i;
    }
    return this->numptl-1;
}

cv::Rect ParticleFilter::applyMotionModel(const cv::Rect &bb) {
    return cv::Rect(bb.x + sampleFromGaussian(0,sigma),bb.y + sampleFromGaussian(0,sigma),bb.width,bb.height);
}
void ParticleFilter::track(cv::Mat &img) {

    this->normalizeWeights();
    particles = resample();
    for(int i=0;i<this->numptl;++i){
        particles[i].bb = this->applyMotionModel(particles[i].bb);

        if(particles[i].bb.x + particles[i].bb.width >= img.cols) particles[i].bb.x = img.cols - 1 - particles[i].bb.width;
        if(particles[i].bb.y + particles[i].bb.height >= img.rows) particles[i].bb.y = img.rows - 1 - particles[i].bb.height;

        if(particles[i].bb.x < 0) particles[i].bb.x = img.cols + particles[i].bb.x - particles[i].bb.width;
        if(particles[i].bb.y < 0) particles[i].bb.y = img.rows + particles[i].bb.y - particles[i].bb.height;

        particles[i].updateParticle(img,particles[i].bb);
        particles[i].measureFitness(this->refhist);
    }

    double maxFitness = - 100000;
    int fittestParticle = -1;
    for(int i=0;i<this->numptl;++i){
        particles[i].fitness *= calLikelihood(particles[i].fitness,sigma);
    }

    for(int i=0;i<this->numptl;++i) {
        if (particles[i].fitness > maxFitness) {
            maxFitness = particles[i].fitness;
            fittestParticle = i;
        }
    }
    //this->showParticle(img,fittestParticle);
    particles[fittestParticle].hist.copyTo(this->refhist);
}


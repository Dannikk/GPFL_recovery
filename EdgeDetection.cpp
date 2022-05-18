//
// Created by nikit on 17.5.22.
//

#include "EdgeDetection.h"
#include <cmath>
#include <iostream>

using namespace std;

void gradient_module(int x, int y, double* gradients, int numExtIter,
                     struct parameters* params, vector<struct chromosome*> chromos, double epsilon){
    int numImages = getNumImages(params);
    double f_lx, f_rx, f_ly, f_ry, dx, dy, b;
    auto *differentials = new double[2*numImages];
    auto *lx = new double [2]{double(x)- epsilon, double(y)};
    auto *rx = new double [2]{double(x)+ epsilon, double(y)};
    auto *ly = new double [2]{double(x), double(y) - epsilon};
    auto *ry = new double [2]{double(x), double(y) + epsilon};

    for(int i=0; i < numImages; i++) {
        gradients[i]=0;
        differentials[2*i]=0;
        differentials[2*i+1]=0;
    }

    for(int ei=0; ei < numExtIter; ei++) {
        executeChromosome(chromos[ei], lx);
        f_lx = getChromosomeOutput(chromos[ei],0);
        executeChromosome(chromos[ei], rx);
        f_rx = getChromosomeOutput(chromos[ei],0);

        executeChromosome(chromos[ei], ly);
        f_ly = getChromosomeOutput(chromos[ei],0);
        executeChromosome(chromos[ei], ry);
        f_ry = getChromosomeOutput(chromos[ei],0);

        for(int i=0; i < numImages; i++){
            b = getB(chromos[ei], i);
            differentials[2*i] += (f_rx - f_lx) * b;
            differentials[2*i+1] += (f_ry - f_ly) * b;
        }

    }

    for(int i=0; i < numImages; i++){
        dx = differentials[2*i] / 2 / epsilon;
        dy = differentials[2*i+1] / 2 / epsilon;
        gradients[i] = sqrt(dx*dx + dy*dy);
    }

    delete [] lx;
    delete [] rx;
    delete [] ly;
    delete [] ry;
    delete [] differentials;

}

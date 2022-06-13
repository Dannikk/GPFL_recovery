//
// Created by nikit on 17.5.22.
//

#include "EdgeDetection.h"
#include <cmath>
#include <iostream>

using namespace std;

void gradient_module(int x, int y, double* gradients, int numExtIter,
                     struct parameters* params, vector<struct chromosome*> chromos, double epsilon, bool print_points){
    int numImages = getNumImages(params);
    double f_lx, f_rx, f_ly, f_ry, dx, dy, a, b;
//    auto *differentials = new double[2*numImages];
    auto* f_left_x = new double[numImages];
    auto* f_right_x = new double[numImages];
    auto* f_left_y = new double[numImages];
    auto* f_right_y = new double[numImages];

    auto *lx = new double [2]{double(x)- epsilon, double(y)};
    auto *rx = new double [2]{double(x)+ epsilon, double(y)};
    auto *ly = new double [2]{double(x), double(y) - epsilon};
    auto *ry = new double [2]{double(x), double(y) + epsilon};

    if (print_points){
        executeChromosome(chromos[0], lx);
        cout << "(" << lx[0] << ", " << lx[1] << ")" << "=" << getChromosomeOutput(chromos[0],0) << endl;

        executeChromosome(chromos[0], rx);
        cout << "(" << rx[0] << ", " << rx[1] << ")" << "=" << getChromosomeOutput(chromos[0],0) << endl;

        executeChromosome(chromos[0], ly);
        cout << "(" << ly[0] << ", " << ly[1] << ")" << "=" << getChromosomeOutput(chromos[0],0) << endl;

        executeChromosome(chromos[0], ry);
        cout << "(" << ry[0] << ", " << ry[1] << ")" << "=" << getChromosomeOutput(chromos[0],0) << endl;
    }

    for(int i=0; i < numImages; i++) {
        gradients[i]=0;
       /* differentials[2*i]=0;
        differentials[2*i+1]=0;*/
        f_left_x[i]=0;
        f_right_x[i]=0;
        f_left_y[i]=0;
        f_right_y[i]=0;
    }

    for(int ei=250; ei < numExtIter; ei++) {
        executeChromosome(chromos[ei], lx);
        f_lx = getChromosomeOutput(chromos[ei],0);
        executeChromosome(chromos[ei], rx);
        f_rx = getChromosomeOutput(chromos[ei],0);

        executeChromosome(chromos[ei], ly);
        f_ly = getChromosomeOutput(chromos[ei],0);
        executeChromosome(chromos[ei], ry);
        f_ry = getChromosomeOutput(chromos[ei],0);

        for(int i=0; i < numImages; i++){
            a = getA(chromos[ei], i);
            b = getB(chromos[ei], i);
            f_left_x[i] += a + b*f_lx;
            f_right_x[i] += a + b*f_rx;
            f_left_y[i] += a + b*f_ly;
            f_right_y[i] += a + b*f_ry;
//            differentials[2*i] += (f_rx - f_lx) * b;
//            differentials[2*i+1] += (f_ry - f_ly) * b;
        }
    }

    for(int i=0; i < numImages; i++){
        f_left_x[i] =  f_left_x[i] < 0  ? 0 : f_left_x[i];
        f_right_x[i] = f_right_x[i] < 0 ? 0 : f_right_x[i];
        f_left_y[i] =  f_left_y[i] < 0  ? 0 : f_left_y[i];
        f_right_y[i] = f_right_y[i] < 0 ? 0 : f_right_y[i];
        dx = (f_right_x[i] - f_left_x[i]) / 2 / epsilon;
        dy = (f_right_y[i] - f_left_y[i]) / 2 / epsilon;
        /*dx = differentials[2*i] / 2 / epsilon;
        dy = differentials[2*i+1] / 2 / epsilon;*/
        gradients[i] = sqrt(dx*dx + dy*dy);
    }

    delete [] lx;
    delete [] rx;
    delete [] ly;
    delete [] ry;
    delete [] f_left_x;
    delete [] f_right_x;
    delete [] f_left_y;
    delete [] f_right_y;
    //delete [] differentials;

}

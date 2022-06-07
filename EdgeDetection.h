//
// Created by nikit on 17.5.22.
//

#include "cgp.h"
#include <vector>

#ifndef GPFL_RECOVERY_EDGEDETECTION_H
#define GPFL_RECOVERY_EDGEDETECTION_H

using namespace std;


void gradient_module(int x, int y, double* gradients, int numExtIter,
                     struct parameters* params, vector<struct chromosome*> chromos, double epsilon, bool print_points);


#endif //GPFL_RECOVERY_EDGEDETECTION_H

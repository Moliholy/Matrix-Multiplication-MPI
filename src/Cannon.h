/*
 * Cannon.h
 *
 *  Created on: Apr 29, 2014
 *      Author: molina
 */

#ifndef CANNON_H_
#define CANNON_H_

#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <cblas.h>
#include <time.h>


using namespace std;


void generateRandomMatrix(double* matrix, int numElements);

void cannon(int p, int pr, int pc, int n,
double *C, int ldC,
double *&A, int ldA,
double *&B, int ldB,
double *comp_time,
double *comm_time);


#endif /* CANNON_H_ */

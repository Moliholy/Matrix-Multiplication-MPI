/*
 * Cannon.cpp
 *
 *  Created on: Apr 29, 2014
 *      Author: molina
 */

#include "Cannon.h"

/**
 * Generates a random matrix with numElements elements
 */
void generateRandomMatrix(double* matrix, int numElements) {
	for (int i = 0; i < numElements; ++i)
		matrix[i] = ((double) rand() / RAND_MAX) * pow(-1.0, (double) rand());
}

/**
 * Returns the ID of the process in the mesh in column-major distribution according to
 * the coordinates provided and the size of the mesh.
 * The mesh has to be of size x size obligatorily.
 */
int getProcessNumber(int row, int column, int size) {
	if (row < 0)
		row += size;
	if (column < 0)
		column += size;
	if (row >= size)
		row -= size;
	if (column >= size)
		column -= size;

	return column * size + row;
}

/**
 * Calls the cannon function to solve a matrix-matrix multiplication problem
 *
 * Parameters:
 * p = number of processes
 * pr = process mesh row coordinate
 * pc = process mesh column coordinate
 * n = (real) size of matrices
 * C = final matrix to calculate
 * ldC = local column stride of C
 * A = one of the matrices in the operation
 * ldA = local column stride of A
 * B = the other matrix for the operation
 * ldB = local column stride of B
 * comp_time = time spent in computation
 * comm_time = time spent in communication
 */
void cannon(int p, int pr, int pc, int n, double* C, int ldC, double* &A,
		int ldA, double* &B, int ldB, double* comp_time, double* comm_time) {

	/* Starting mesurements */
	clock_t t = clock();
	int numLocalElements = n / p;
	MPI_Request sreq_left, sreq_top;
	MPI_Status rreq_bottom, rreq_right;
	//MPI_Status rreq_right, rreq_bottom;
	int maxSteps = sqrt((double) p);

	//to exchange buffers
	double* aux;
	double* auxA = new double[numLocalElements];
	double* auxB = new double[numLocalElements];

	int left = getProcessNumber(pr, pc - 1, maxSteps);
	int right = getProcessNumber(pr, pc + 1, maxSteps);
	int top = getProcessNumber(pr - 1, pc, maxSteps);
	int bottom = getProcessNumber(pr + 1, pc, maxSteps);

	//for the first receiving
	int firstRecvA = getProcessNumber(pr, (pr + pc) % p, maxSteps);
	int firstRecvB = getProcessNumber((pr + pc) % p, pc, maxSteps);

	//for the first sending
	int distA = ((pr + pc) % p) - pc;
	distA = distA < 0 ? distA + maxSteps : distA;
	int firstSendA = getProcessNumber(pr, pc - distA, maxSteps);

	int distB = ((pr + pc) % p) - pr;
	distB = distB < 0 ? distB + maxSteps : distB;
	int firstSendB = getProcessNumber(pr - distB, pc, maxSteps);

	// sending my information of A
	MPI_Isend(A, numLocalElements, MPI_DOUBLE, firstSendA, 0,
	MPI_COMM_WORLD, &sreq_left);

	//receiving the information about A
	MPI_Recv(auxA, numLocalElements, MPI_DOUBLE, firstRecvA, 0,
	MPI_COMM_WORLD, &rreq_right);

	//sending my information of B
	MPI_Isend(B, numLocalElements, MPI_DOUBLE, firstSendB, 0,
	MPI_COMM_WORLD, &sreq_top);

	//receiving the information about B
	MPI_Recv(auxB, numLocalElements, MPI_DOUBLE, firstRecvB, 0,
	MPI_COMM_WORLD, &rreq_bottom);

	//we have to block after the sending operation!
	/* Note: for big values of n it starts to fail. It is necessary to determine
	 * when the sending operation has finished. Otherwise the buffer is read and written
	 * at the same time, producing errors that have been empirically found out looking at the
	 * residual norm once the program was completed.
	 */
	MPI_Wait(&sreq_left, MPI_STATUS_IGNORE);
	MPI_Wait(&sreq_top, MPI_STATUS_IGNORE);

	//swapping buffers
	aux = A;
	A = auxA;
	auxA = aux;

	aux = B;
	B = auxB;
	auxB = aux;

	/* Finishing measurements */
	*comm_time = clock() - t;
	*comp_time = 0.0;

	for (int step = 0; step < maxSteps; ++step) {
		t = clock();
		int size = sqrt((double) numLocalElements);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size, size, size,
				1.0, A, ldA, B, ldB, 1.0, C, ldC);
		*comp_time += clock() - t;

		if (step < maxSteps - 1) {
			t = clock();

			// sending my information of A
			MPI_Isend(A, numLocalElements, MPI_DOUBLE, left, 0, MPI_COMM_WORLD,
					&sreq_left);
			//receiving the information about A
			MPI_Recv(auxA, numLocalElements, MPI_DOUBLE, right, 0,
			MPI_COMM_WORLD, &rreq_right);

			//sending my information of B
			MPI_Isend(B, numLocalElements, MPI_DOUBLE, top, 0, MPI_COMM_WORLD,
					&sreq_top);
			//receiving the information about B
			MPI_Recv(auxB, numLocalElements, MPI_DOUBLE, bottom, 0,
			MPI_COMM_WORLD, &rreq_bottom);

			//waiting for the send to finish
			MPI_Wait(&sreq_left, MPI_STATUS_IGNORE);
			MPI_Wait(&sreq_top, MPI_STATUS_IGNORE);

			//swapping buffers
			aux = A;
			A = auxA;
			auxA = aux;

			aux = B;
			B = auxB;
			auxB = aux;

			*comm_time += clock() - t;
		}
	}
	delete[] auxA;
	delete[] auxB;

	*comp_time /= ((double) CLOCKS_PER_SEC);
	*comm_time /= ((double) CLOCKS_PER_SEC);
}

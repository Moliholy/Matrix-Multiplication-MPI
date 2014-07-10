/*
 ============================================================================
 Name        : DAPA_ass1.c
 Author      : Jose Molina
 Version     :
 Copyright   : Your copyright notice
 Description : Compute Pi in MPI C++
 ============================================================================
 */

#include <mpi.h>
#include <iostream>
#include "Cannon.h"
#include <cblas.h>
using namespace std;

int me, np, n;

void reorderMatrix(double* &C, int localElements, int size, int subsize) {
	double *Cr = new double[n];

	int count = 0;
	int magicNumber = size / subsize;
	int rep = sqrt((double) np);
	int currentAdress = 0;
	int increment = 0;
	for (int i = 0; i < rep; i++) {
		currentAdress = i * magicNumber * magicNumber * rep;
		for (int j = 0; j < rep * magicNumber; j++) {
			for (int k = 0; k < magicNumber; k++)
				Cr[count++] = C[currentAdress++];
			if (j % rep != rep - 1)
				increment = magicNumber * (magicNumber - 1);
			else
				increment = -(magicNumber * magicNumber) * (rep - 1);
			currentAdress += increment;
		}
	}
	delete[] C;
	C = Cr;
}

void printMatrix(double* matrix, int size) {
	cout.setf(ios::fixed, ios::floatfield);
	for (int i = 0; i < size; ++i) {
		cout << endl;
		for (int j = 0; j < size; ++j) {
			cout << matrix[j * size + i] << "\t";
		}
	}
	cout << endl;
}

double* checkResult(double* localC) {
	double* C = 0;
	if (me == 0) {
		C = new double[n];
	}

	MPI_Gather(localC, n / np, MPI_DOUBLE, C, n / np, MPI_DOUBLE, 0,
	MPI_COMM_WORLD);
	if (me == 0) {
		reorderMatrix(C, n / np, sqrt((double) n), sqrt((double) np));
	}
	return C;
}

double* checkOperation(double* localA, double* localB) {
	double* C = 0, *A = 0, *B = 0;
	if (me == 0) {
		C = new double[n];
		A = new double[n];
		B = new double[n];
		for (int i = 0; i < n; ++i)
			C[i] = 0.0;
	}
	int localElements = n / np;

	MPI_Gather(localA, localElements, MPI_DOUBLE, A, localElements, MPI_DOUBLE,
			0,
			MPI_COMM_WORLD);

	MPI_Gather(localB, localElements, MPI_DOUBLE, B, localElements, MPI_DOUBLE,
			0,
			MPI_COMM_WORLD);

	if (me == 0) {
		int size = sqrt((double) n);
		int subsize = sqrt((double) np);

		//printMatrix(A, size);

		reorderMatrix(A, localElements, size, subsize);
		reorderMatrix(B, localElements, size, subsize);

		//printMatrix(A, size);

		//matrixMultiplication(C, A, B, size);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size, size, size,
				1.0, A, size, B, size, 1.0, C, size);

		//printMatrix(C, size);

		delete[] A;
		delete[] B;
	}
	return C;
}

/**
 * Computes the residual norm
 */
double computeResidualNorm(double *serialC, double* parallelC) {
	double norm = 0.0;
	for (int i = 0; i < n; ++i) {
		double num = serialC[i] - parallelC[i];
		norm += num * num;
	}
	return sqrt((double) norm);
}

/**
 * Usage:
 * ./program size
 *
 * every process will deal with a size x size sub-matrix
 */
int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	int size;
	bool check;
	int repetitions = 1;

	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	int div = (int) sqrt((double) np);

	if (argc < 2) {
		cout << "Size not specified" << endl;
		return -1;
	} else if (argc == 2) {
		size = atoi(argv[1]) / div;
		repetitions = 1;
		check = false;
	} else if (argc == 3) {
		check = strcmp((const char*) argv[1], "-p") == 0;
		if (check) {
			size = atoi(argv[2]) / div;
			repetitions = 1;
		} else {
			size = atoi(argv[1]) / div;
			repetitions = atoi(argv[2]);
		}
	} else {
		check = strcmp((const char*) argv[1], "-p") == 0;
		size = atoi(argv[2]);
		repetitions = atoi(argv[3]);
	}

	n = np * size * size;

	srand(time(NULL) + me);

	int numElements = size * size;
//generating C
	double *C = new double[numElements];
	for (int i = 0; i < numElements; ++i)
		C[i] = 0.0;

//generating A
	double *A = new double[numElements];
	generateRandomMatrix(A, numElements);

//generating B
	double *B = new double[numElements];
	generateRandomMatrix(B, numElements);

	double* Ccheck = 0;
	if (check)
		Ccheck = checkOperation(A, B);

//we are going to use a column-major distribution of processes
	/*
	 * For example, for 16 processors
	 * 0	4	8	12
	 * 1	5	9	13
	 * 2	6	10	14
	 * 3	7	11	15
	 */
	int dist = (int) sqrt((double) np);
	int rowCoordinate = me % dist;
	int columnCoordinate = me / dist;
	double comm_time = 0.0, comp_time = 0.0;

	//starting repetitions
	double* copyA = new double[numElements];
	double* copyB = new double[numElements];
	double* copyC = new double[numElements];

	for (int i = 0; i < repetitions; ++i) {
		/*memcpy(copyA, A, numElements * sizeof(double));
		 memcpy(copyB, B, numElements * sizeof(double));*/
		for (int j = 0; j < numElements; ++j) {
			copyA[j] = A[j];
			copyB[j] = B[j];
			copyC[j] = 0.0;
		}

		comm_time = 0.0;
		comp_time = 0.0;
		//previous synchronization
		MPI_Barrier(MPI_COMM_WORLD);

		//measuring the time
		clock_t t = clock();

		//calling the cannon function
		cannon(np, rowCoordinate, columnCoordinate, n, copyC, size, copyA, size,
				copyB, size, &comp_time, &comm_time);

		
		//after finishing ALL processes we masure the time
		MPI_Barrier(MPI_COMM_WORLD);
		double parallelExecutionTime = ((double) (clock() - t))
				/ ((double) (CLOCKS_PER_SEC));

		double sumCommTime = 0.0, sumCompTime = 0.0, minParallelExecutionTime =
				0.0;

		//gathering the MINIMUM parallel time
		MPI_Reduce(&parallelExecutionTime, &minParallelExecutionTime, 1,
		MPI_DOUBLE,
		MPI_MIN, 0,
		MPI_COMM_WORLD);

		//gathering communication time firstly
		MPI_Reduce(&comm_time, &sumCommTime, 1, MPI_DOUBLE,
		MPI_SUM, 0,
		MPI_COMM_WORLD);

		//gathering computation time secondly
		MPI_Reduce(&comp_time, &sumCompTime, 1, MPI_DOUBLE,
		MPI_SUM, 0,
		MPI_COMM_WORLD);

		double* Cresult = 0;
		if (check)
			Cresult = checkResult(copyC);

		//printing results
		if (me == 0) {
			int originalInput = size * div;
			if (check) {
				double resNorm = computeResidualNorm(Ccheck, Cresult);
				cout << originalInput << " " << minParallelExecutionTime << " "
						<< sumCommTime << " " << sumCompTime << " " << resNorm
						<< endl;
			} else {
				cout << originalInput << " " << minParallelExecutionTime << " "
						<< sumCommTime << " " << sumCompTime << endl;
			}
		}
		delete[] Cresult;
	}

//printResult(C);
	delete[] A;
	delete[] B;
	delete[] C;
	delete[] copyA;
	delete[] copyB;
	delete[] copyC;
	delete[] Ccheck;
	MPI_Finalize();

	return 0;
}

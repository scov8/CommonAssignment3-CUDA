/*
* Course: High Performance Computing 2021/2022
*
* Lecturer: Francesco Moscato	fmoscato@unisa.it
*
* Group:
* Rosa Gerardo	    0622701829	g.rosa10@studenti.unisa.it
* Scovotto Luigi    0622701702  l.scovotto1@studenti.unisa.it
* Tortora Francesco 0622701700  f.tortora21@studenti.unisa.it
*
* Copyright (C) 2022 - All Rights Reserved
* This file is part of CommonAssignment3.
*
* Requirements: Parallelize and Evaluate Performances of "COUNTING SORT" Algorithm ,by using CUDA.
*
* The previous year's group 02 files proposed by the professor during the course were used for file generation and extraction.
*
* CommonAssignment3 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* CommonAssignment3 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with CommonAssignment3.  If not, see <http://www.gnu.org/licenses/>.
*
* You can find the complete project on GitHub:
* https://github.com/scov8/CommonAssignment3
*/

#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#define DATATYPE int
#define MAX_VALUE 100
#define CUDA_CHECK(X)                                                     \
    {                                                                     \
        cudaError_t _m_cudaStat = X;                                      \
        if (cudaSuccess != _m_cudaStat)                                   \
        {                                                                 \
            fprintf(stderr, "\nCUDA_ERROR: %s in file %s line %d\n",      \
                    cudaGetErrorString(_m_cudaStat), __FILE__, __LINE__); \
            exit(1);                                                      \
        }                                                                 \
    }

/**
 * @brief                 Reports the data for analysis in a csv file
 *
 * @param time            execution time.
 * @param N               size of array.
 * @param gridsize        number of blocks in a grid.
 * @param THREADxBLOCK    number of threads in a block.
 */
void make_csv(float time, int N, int gridsize, int THREADxBLOCK)
{
    FILE *fp;
    char root_filename[] = "shared_measures";

    char *filename = (char *)malloc(sizeof(char) * (strlen(root_filename) + 10 * sizeof(char)));
    sprintf(filename, "%s_%d.csv", root_filename, THREADxBLOCK);

    if (access(filename, F_OK) == 0)
    {
        fp = fopen(filename, "a");
    }
    else
    {
        fp = fopen(filename, "w");
        fprintf(fp, "N, BlockSize, GridSize, time_sec\n");
    }
    fprintf(fp, "%d, %d, %d, %f\n", N, THREADxBLOCK, gridsize, time);
    fclose(fp);
}

/**
 * @brief            Apply the histogram phase of the counting sort on the Kernel.
 *
 * @param A          a pointer to an array which must be sorted.
 * @param C          a pointer to a count array.
 * @param N          size of array.
 */
__global__ void countingSortKernel1(DATATYPE *A, DATATYPE *C, int N, int K)
{
    __shared__ DATATYPE sharedCount[MAX_VALUE];

    int tid = threadIdx.x;

    if (tid < K)
        sharedCount[tid] = 0;

    __syncthreads();

    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    while (index < N)
    {
        atomicAdd(&sharedCount[A[index]], 1);
        index += stride;
    }
    __syncthreads();

    if (tid < K)
        atomicAdd(&C[tid], sharedCount[tid]);

    return;
}

/**
 * @brief     Transform the frequencies in the count array into indices on the Kernel.
 *
 * @param C   a pointer to a count array.
 * @param K   max value of array.
 */
__global__ void countingSortKernel2(DATATYPE *C, int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        for (int i = 1; i < K; i++)
            C[i] += C[i - 1];
    }
}

/**
 * @brief            Distribute the numbers from the input array into the output array in sorted order on the Kernel.
 *
 * @param A          a pointer to an array which must be sorted.
 * @param B          a pointer to a result array.
 * @param C          a pointer to a count array.
 * @param N          size of array.
 */
__global__ void countingSortKernel3(DATATYPE *A, DATATYPE *B, DATATYPE *C, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride)
        B[atomicSub(&C[A[i]], 1) - 1] = A[i];
}

/**
 * @brief                   Transform the frequencies in the count array into indices on the Kernel.
 *
 * @param initArray         a pointer to an array which must be sorted.
 * @param outputArray       a pointer to a result array.
 * @param maxValue          max value of array.
 * @param length            size of array.
 * @param THREADxBLOCK      number of threads in a block.
 */
void countingSortOnDevice(DATATYPE *initArray, DATATYPE *outputArray, int maxValue, int length, int THREADxBLOCK)
{
    int sizeArray = length * sizeof(DATATYPE);
    //float mflops;
    DATATYPE *device_initArray, *device_outputArray, *count;

    dim3 gridSize((length - 1) / THREADxBLOCK + 1);
    dim3 blockSize(THREADxBLOCK);
    printf("GridSize: %d\n", gridSize.x);

    CUDA_CHECK(cudaMalloc((void **)&device_initArray, sizeArray));
    CUDA_CHECK(cudaMemcpy(device_initArray, initArray, sizeArray, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&count, MAX_VALUE * sizeof(DATATYPE)));
    CUDA_CHECK(cudaMemset(count, 0, MAX_VALUE * sizeof(DATATYPE)));

    CUDA_CHECK(cudaMalloc((void **)&device_outputArray, sizeArray));

    cudaError_t mycudaerror;
    mycudaerror = cudaGetLastError();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    countingSortKernel1<<<gridSize, blockSize>>>(device_initArray, count, length, maxValue);

    gridSize.x = (MAX_VALUE - 1) / THREADxBLOCK + 1;
    countingSortKernel2<<<gridSize, blockSize>>>(count, maxValue);

    gridSize.x = (length - 1) / THREADxBLOCK + 1;
    countingSortKernel3<<<gridSize, blockSize>>>(device_initArray, device_outputArray, count, length);

    mycudaerror = cudaGetLastError();
    if (mycudaerror != cudaSuccess)
    {
        fprintf(stderr, "%s\n", cudaGetErrorString(mycudaerror));
        exit(1);
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed = elapsed / 1000.f; // convert to seconds
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("Kernel elapsed time %fs \n", elapsed);
    
    make_csv(elapsed, length, gridSize.x, THREADxBLOCK);

    CUDA_CHECK(cudaMemcpy(outputArray, device_outputArray, sizeArray, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_initArray));
    CUDA_CHECK(cudaFree(device_outputArray));
    CUDA_CHECK(cudaFree(count));
}

int main(int argc, char **argv)
{
    DATATYPE *initArray, *host_outputArray, *device_outputArray;

    if (argc < 3)
    {
        fprintf(stderr, "Enter length and thread number per block\n");
        exit(1);
    }
    int length = atoi(argv[1]);
    int THREADxBLOCK = atoi(argv[2]);

    if (length < 1)
    {
        fprintf(stderr, "Error length=%d, must be > 0\n", length);
        exit(1);
    }
    if (THREADxBLOCK > 1024)
    {
        fprintf(stderr, "Error THREADxBLOCK=%d, must be < 1024\n", THREADxBLOCK);
        exit(1);
    }

    initArray = (DATATYPE *)calloc(length, sizeof(DATATYPE));
    host_outputArray = (DATATYPE *)calloc(length, sizeof(DATATYPE));
    device_outputArray = (DATATYPE *)calloc(length, sizeof(DATATYPE));

    if (initArray == NULL)
    {
        fprintf(stderr, "Could not get memory for initArray\n");
        exit(1);
    }
    if (host_outputArray == NULL)
    {
        fprintf(stderr, "Could not get memory for host_outputArray\n");
        exit(1);
    }
    if (device_outputArray == NULL)
    {
        fprintf(stderr, "Could not get memory for device_outputArray\n");
        exit(1);
    }

    unsigned int seed = time(NULL);
    for (int i = 0; i < length; i++)
        initArray[i] = rand_r(&seed) % MAX_VALUE;

    countingSortOnDevice(initArray, device_outputArray, MAX_VALUE, length, THREADxBLOCK);
}

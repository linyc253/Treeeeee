#include <stdio.h>
#include <stdlib.h>
#include "parameter.h"
#include "tree.h"
#include "particle.h"

__device__ float3 two_partical_force_gpu(float4 a, float4 b, float3 f, float epsilon){
    float3 dx;
    float r_sq = 0.0;
    dx.x = a.x - b.x;
    dx.y = a.y - b.y;
    dx.z = a.z - b.z;
    r_sq = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
    float F_mag = -(a.w * b.w) / pow(r_sq + epsilon*epsilon, 1.5);
    f.x += F_mag * dx.x;
    f.y += F_mag * dx.y;
    f.z += F_mag * dx.z;
    return f;
}

__global__ void Particle_Cell_Kernel(float4* P, float4* C, int nl, float3* F, float epsilon) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int ng = blockDim.x;

    extern __shared__ float4 shPosition[];
    float4 myPosition = P[threadIdx.x];

    int i = id;
    float3 f = {0.0, 0.0, 0.0};
    while(i < nl + ng){
        if(i < ng) shPosition[threadIdx.x] = P[i];
        else shPosition[threadIdx.x] = C[i - ng];
        __syncthreads();

        for (int j = 0; j < blockDim.x; j++) {
            //if(j == i && i < ng) continue;
            f = two_partical_force_gpu(myPosition, shPosition[j], f, epsilon);
        }
        __syncthreads();

        i += blockDim.x * gridDim.x;
    }

    F[id] = f;
}

__global__ void Force_Reduction_Kernel(float3* F){
    int ib = blockDim.x;
    int ng = gridDim.x;
    while (ib != 0) {
        if(threadIdx.x < ib){
            F[threadIdx.x * ng + blockIdx.x].x += F[(threadIdx.x + ib) * ng + blockIdx.x].x;
            F[threadIdx.x * ng + blockIdx.x].y += F[(threadIdx.x + ib) * ng + blockIdx.x].y;
            F[threadIdx.x * ng + blockIdx.x].z += F[(threadIdx.x + ib) * ng + blockIdx.x].z;
        }
        __syncthreads();

        ib /=2;
    }
}

extern "C" void Particle_Cell_Force_gpu(Coord4* P, int ng, Coord4* C, int nl, Coord3* F, double epsilon){
    float4 *d_P, *d_C;
    float3 *d_F;

    int threadsPerBlock = ng;
    int blocksPerGrid = 1;

    // Allocate device memory
    cudaMalloc((void**)&d_P, sizeof(float4) * ng);
    cudaMalloc((void**)&d_C, sizeof(float4) * nl);
    cudaMalloc((void**)&d_F, sizeof(float3) * ng * blocksPerGrid);

    // Transfer data from host to device memory
    cudaMemcpy(d_P, P, sizeof(float4) * ng, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(float4) * nl, cudaMemcpyHostToDevice);

    // Executing kernel
    int sm = threadsPerBlock*sizeof(float4);
    Particle_Cell_Kernel<<<blocksPerGrid, threadsPerBlock, sm>>>(d_P, d_C, nl, d_F, (float)epsilon);
    cudaDeviceSynchronize();
    Force_Reduction_Kernel<<<threadsPerBlock, blocksPerGrid / 2>>>(d_F);
    cudaDeviceSynchronize();

    // Transfer data back to host memory
    cudaMemcpy(F, d_F, sizeof(float3) * ng, cudaMemcpyDeviceToHost);
    //printf("%f %f\n", ((float3*)F)->x, F->x[0]);

    // Deallocate device memory
    cudaFree(d_P);
    cudaFree(d_C);
    cudaFree(d_F);

    return;
}

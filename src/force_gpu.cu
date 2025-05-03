#include <stdio.h>
#include <stdlib.h>
#include "parameter.h"
#include "tree.h"
#include "particle.h"

// Not optimized (too many global memory access)
__global__ void Particle_Cell_Kernel(float4* P, int ng, float4* C, int nl, float3* F, float epsilon) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    // Particle-Cell
    if(id < ng * nl){
        int ic = id / ng;
        int ip = id % ng;
        float3 dx;
        float r_sq = 0.0;
        dx.x = P[ip].x - C[ic].x;
        dx.y = P[ip].y - C[ic].y;
        dx.z = P[ip].z - C[ic].z;
        r_sq = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
        float F_mag = -(P[ip].w * C[ic].w) / pow(r_sq + epsilon*epsilon, 1.5);
        F[ip].x += F_mag * dx.x;
        F[ip].y += F_mag * dx.y;
        F[ip].z += F_mag * dx.z;
    }
    // Particle-Particle
    else if(id < ng * (nl + ng)){
        id -= ng * nl;
        int ipp = id / ng;
        int ip = id % ng;
        float3 dx;
        float r_sq = 0.0;
        dx.x = P[ip].x - P[ipp].x;
        dx.y = P[ip].y - P[ipp].y;
        dx.z = P[ip].z - P[ipp].z;
        r_sq = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
        float F_mag = -(P[ip].w * P[ipp].w) / pow(r_sq + epsilon*epsilon, 1.5);
        F[ip].x += F_mag * dx.x;
        F[ip].y += F_mag * dx.y;
        F[ip].z += F_mag * dx.z;
    }
    __syncthreads();
}

extern "C" void Particle_Cell_Force_gpu(Coord4* P, int ng, Coord4* C, int nl, Coord3* F, double epsilon){
    float4 *d_P, *d_C;
    float3 *d_F;

    // Allocate device memory
    cudaMalloc((void**)&d_P, sizeof(float4) * ng);
    cudaMalloc((void**)&d_C, sizeof(float4) * nl);
    cudaMalloc((void**)&d_F, sizeof(float3) * ng);

    // Transfer data from host to device memory
    cudaMemcpy(d_P, P, sizeof(float4) * ng, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(float4) * nl, cudaMemcpyHostToDevice);

    // Initialize d_F as 0.0 (https://forums.developer.nvidia.com/t/can-i-set-a-floats-to-zero-with-cudamemset/153706)
    cudaMemset(d_F, 0, sizeof(float3) * ng);

    // Executing kernel
    int threadsPerBlock = 1;
    int blocksPerGrid = (ng * (nl + ng) + threadsPerBlock - 1) / threadsPerBlock; // ceil(ng * (nl + ng) / threadsPerBlock)
    Particle_Cell_Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_P, ng, d_C, nl, d_F, (float)epsilon);
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

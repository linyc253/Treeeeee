#include <stdio.h>
#include <stdlib.h>
#include "parameter.h"
#include "tree.h"
#include "particle.h"

__global__ void Particle_Cell_Kernel(float4* P, int ng, float4* C, int nl, float3* F, float epsilon) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int ip = id % ng;
    float3 f = {0.0, 0.0, 0.0};
    float4 p = P[ip];
    while(id < ng * (nl + ng)){
        // Particle-Cell
        if(id < ng * nl){
            int ic = id / ng;
            float3 dx;
            dx.x = p.x - C[ic].x;
            dx.y = p.y - C[ic].y;
            dx.z = p.z - C[ic].z;
            float inv_r = rsqrtf(dx.x * dx.x + dx.y * dx.y + dx.z * dx.z + epsilon * epsilon);
            float F_mag = -(p.w * C[ic].w) * inv_r * inv_r * inv_r;
            f.x += F_mag * dx.x;
            f.y += F_mag * dx.y;
            f.z += F_mag * dx.z;
        }
        // Particle-Particle
        else if(id < ng * (nl + ng)){
            int ipp = id / ng - nl;
            float3 dx;
            dx.x = p.x - P[ipp].x;
            dx.y = p.y - P[ipp].y;
            dx.z = p.z - P[ipp].z;
            float inv_r = rsqrtf(dx.x * dx.x + dx.y * dx.y + dx.z * dx.z + epsilon * epsilon);
            float F_mag = -(p.w * P[ipp].w) * inv_r * inv_r * inv_r;
            f.x += F_mag * dx.x;
            f.y += F_mag * dx.y;
            f.z += F_mag * dx.z;
        }
        id += blockDim.x * gridDim.x;
    }
    atomicAdd(&F[ip].x, f.x);
    atomicAdd(&F[ip].y, f.y);
    atomicAdd(&F[ip].z, f.z);
}

__global__ void Particle_Cell_Kernel_Potential(float4* P, int ng, float4* C, int nl, float3* F, float epsilon) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int ip = id % ng;
    float3 f = {0.0, 0.0, 0.0};
    float4 p = P[ip];
    while(id < ng * (nl + ng)){
        // Particle-Cell
        if(id < ng * nl){
            int ic = id / ng;
            float3 dx;
            dx.x = p.x - C[ic].x;
            dx.y = p.y - C[ic].y;
            dx.z = p.z - C[ic].z;
            float inv_r = rsqrtf(dx.x * dx.x + dx.y * dx.y + dx.z * dx.z + epsilon * epsilon);
            f.x += -(p.w * C[ic].w) * inv_r;
        }
        // Particle-Particle
        else if(id < ng * (nl + ng)){
            int ipp = id / ng - nl;
            float3 dx;
            dx.x = p.x - P[ipp].x;
            dx.y = p.y - P[ipp].y;
            dx.z = p.z - P[ipp].z;
            float inv_r = rsqrtf(dx.x * dx.x + dx.y * dx.y + dx.z * dx.z + epsilon * epsilon);
            if(ipp != ip) f.x += -(p.w * P[ipp].w) * inv_r;
        }
        id += blockDim.x * gridDim.x;
    }
    atomicAdd(&F[ip].x, f.x);
}

extern "C" double Particle_Cell_Force_gpu(InteractionList particle_list, InteractionList cell_list, Coord3* force_xyz, double epsilon, int threadsPerBlock, int compute_energy){
    int ng = particle_list.n;
    int nl = cell_list.n;
    float4 *d_P, *d_C;
    float3 *d_F;
    double V = 0.0;

    // Allocate device memory
    cudaMalloc((void**)&d_P, sizeof(float4) * ng);
    cudaMalloc((void**)&d_C, sizeof(float4) * nl);
    cudaMalloc((void**)&d_F, sizeof(float3) * ng);

    // Transfer data from host to device memory
    cudaMemcpy(d_P, particle_list.xyzm, sizeof(float4) * ng, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, cell_list.xyzm, sizeof(float4) * nl, cudaMemcpyHostToDevice);

    // Initialize d_F as 0.0 (https://forums.developer.nvidia.com/t/can-i-set-a-floats-to-zero-with-cudamemset/153706)
    cudaMemset(d_F, 0, sizeof(float3) * ng);

    // Executing kernel
    int blocksPerGrid = ng;
    if(compute_energy){
        Particle_Cell_Kernel_Potential<<<blocksPerGrid, threadsPerBlock>>>(d_P, ng, d_C, nl, d_F, (float)epsilon);
    }
    else{
        Particle_Cell_Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_P, ng, d_C, nl, d_F, (float)epsilon);
    }
    

    if(cudaSuccess != cudaDeviceSynchronize()){
        printf("CUDA Error!!!\n");
        exit(EXIT_FAILURE);
    }

    // Transfer data back to host memory
    cudaMemcpy(force_xyz, d_F, sizeof(float3) * ng, cudaMemcpyDeviceToHost);

    if(compute_energy){
        for(int i = 0; i < ng; i++){
            V += (double)force_xyz[i].x[0];
        }
    }

    // Deallocate device memory
    cudaFree(d_P);
    cudaFree(d_C);
    cudaFree(d_F);

    return V;
}

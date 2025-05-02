#include <stdio.h>
#include <stdlib.h>
#include <parameter.h>

// Not optimized (too many global memory access, P[ip * (1 + dim) + i] is NOT necessary)
__global__ void Particle_Cell_Kernel(float* P, int ng, float* C, int nl, float* F, float epsilon, int dim) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    // Particle-Cell
    if(id < ng * nl){
        int ic = id / nl;
        int ip = id % nl;
        float dx[dim];
        float r_sq = 0.0;
        for (int i = 0; i < dim; i++){
            dx[i] = P[ip * (1 + dim) + i] - C[ic * (1 + dim) + i];
            r_sq += dx[i] * dx[i];
        }
        float F_mag = -(P[ip * (1 + dim) + dim] * C[ic * (1 + dim) + dim]) / pow(r_sq + epsilon*epsilon, 1.5);
        for (int i = 0; i < dim; i++){
            F[ip * dim + i] += F_mag * dx[i];
        }
    }
    // Particle-Particle
    else if(id < ng * (nl + ng)){
        int ipp = id / ng;
        int ip = id % ng;
        float dx[dim];
        float r_sq = 0.0;
        for (int i = 0; i < dim; i++){
            dx[i] = P[ip * (1 + dim) + i] - P[ipp * (1 + dim) + i];
            r_sq += dx[i] * dx[i];
        }
        float F_mag = -(P[ip * (1 + dim) + dim] * P[ipp * (1 + dim) + dim]) / pow(r_sq + epsilon*epsilon, 1.5);
        for (int i = 0; i < dim; i++){
            F[ip * dim + i] += F_mag * dx[i];
        }
    }
}

void Particle_Cell_Force_gpu(float* P, int ng, float* C, int nl, float* F, double epsilon){
    float *d_P, *d_C, *d_F;

    // Allocate device memory
    cudaMalloc((void**)&d_P, sizeof(float) * ng * (DIM + 1));
    cudaMalloc((void**)&d_C, sizeof(float) * nl * (DIM + 1));
    cudaMalloc((void**)&d_F, sizeof(float) * ng * DIM);

    // Transfer data from host to device memory
    cudaMemcpy(d_P, P, sizeof(float) * ng * (DIM + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(float) * nl * (DIM + 1), cudaMemcpyHostToDevice);

    // Initialize d_F as 0.0 (https://forums.developer.nvidia.com/t/can-i-set-a-floats-to-zero-with-cudamemset/153706)
    cudaMemset(d_F, 0, sizeof(float) * ng * DIM);

    // Executing kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (ng * (nl + ng) + threadsPerBlock - 1) / threadsPerBlock; // ceil(ng * (nl + ng) / threadsPerBlock)
    Particle_Cell_Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_P, ng, d_C, nl, d_F, (float)epsilon, DIM);
    cudaDeviceSynchronize();

    // Transfer data back to host memory
    cudaMemcpy(F, d_F, sizeof(float) * ng * DIM, cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_P);
    cudaFree(d_C);
    cudaFree(d_F);

    return;
}

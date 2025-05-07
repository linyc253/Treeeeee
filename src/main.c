#include <stdio.h>
#include <stdlib.h>
#include "evolution.h"
#include "parameter.h"
#include "particle.h"
#include <sys/time.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef USE_OPENCL
#include <OpenCL/opencl.h>

typedef struct {
    float x, y, z, w;
} float4;

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_mem d_pos, d_acc;

void CheckCLError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        printf("OpenCL Error at %s: %d\n", msg, err);
        exit(1);
    }
}

void Setup_OpenCL(const char* kernel_file, int N) {
    printf("[DEBUG] Entering Setup_OpenCL with N = %d\n", N);
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    err = clGetPlatformIDs(1, &platform, NULL);
    CheckCLError(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CheckCLError(err, "clGetDeviceIDs");

    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using OpenCL device: %s\n", device_name);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CheckCLError(err, "clCreateContext");

    queue = clCreateCommandQueue(context, device, 0, &err);
    CheckCLError(err, "clCreateCommandQueue");

    // ===== Hardcoded absolute path (adjust to your machine) =====
    const char* absolute_kernel_path = "/TREEEEEE/src/force_gpu.cl";
    FILE* fp = fopen(absolute_kernel_path, "r");
    if (!fp) {
        printf("Error: Cannot open kernel file %s\n", absolute_kernel_path);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char* source = malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    CheckCLError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log:\n%s\n", log);
        free(log);
        CheckCLError(err, "clBuildProgram");
    }

    kernel = clCreateKernel(program, "compute_force", &err);
    CheckCLError(err, "clCreateKernel");

    d_pos = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float4) * N, NULL, &err);
    CheckCLError(err, "clCreateBuffer d_pos");

    d_acc = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float4) * N, NULL, &err);
    CheckCLError(err, "clCreateBuffer d_acc");

    free(source);
    printf("[DEBUG] Completed Setup_OpenCL\n");
}

void Cleanup_OpenCL() {
    printf("[DEBUG] Cleaning up OpenCL resources\n");
    clReleaseMemObject(d_pos);
    clReleaseMemObject(d_acc);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void Compute_Force_OpenCL(Particle* P, int N, float eps2) {
    printf("[DEBUG] Entering Compute_Force_OpenCL with N = %d\n", N);

    if (P == NULL) {
        printf("Error: Particle array is NULL!\n");
        exit(1);
    }

    float4* h_pos = malloc(sizeof(float4) * N);
    float4* h_acc = malloc(sizeof(float4) * N);

    if (h_pos == NULL || h_acc == NULL) {
        printf("Error: malloc failed for host buffers!\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        h_pos[i].x = P[i].x[0];
        h_pos[i].y = P[i].x[1];
        h_pos[i].z = P[i].x[2];
        h_pos[i].w = P[i].m;
    }

    cl_int err;
    err = clEnqueueWriteBuffer(queue, d_pos, CL_TRUE, 0, sizeof(float4) * N, h_pos, 0, NULL, NULL);
    CheckCLError(err, "clEnqueueWriteBuffer");

    int Nval = N;
    err = clSetKernelArg(kernel, 0, sizeof(int), &Nval);
    CheckCLError(err, "clSetKernelArg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_pos);
    CheckCLError(err, "clSetKernelArg 1");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_acc);
    CheckCLError(err, "clSetKernelArg 2");

    err = clSetKernelArg(kernel, 3, sizeof(float), &eps2);
    CheckCLError(err, "clSetKernelArg 3");

    size_t global = N;
    printf("[DEBUG] Launching kernel with global size %zu\n", global);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    CheckCLError(err, "clEnqueueNDRangeKernel");

    err = clFinish(queue);
    CheckCLError(err, "clFinish");

    err = clEnqueueReadBuffer(queue, d_acc, CL_TRUE, 0, sizeof(float4) * N, h_acc, 0, NULL, NULL);
    CheckCLError(err, "clEnqueueReadBuffer");

    for (int i = 0; i < N; i++) {
        P[i].f[0] = h_acc[i].x;
        P[i].f[1] = h_acc[i].y;
        P[i].f[2] = h_acc[i].z;
    }

    free(h_pos);
    free(h_acc);
    printf("[DEBUG] Completed Compute_Force_OpenCL\n");
}
#endif

int main() {
    Read_Input_Parameter("Input_Parameter.ini");

    struct stat st = {0};
    const char* OUTDIR = get_string("BasicSetting.OUTDIR", ".");
    if (stat(OUTDIR, &st) == -1) {
        mkdir(OUTDIR, 0700);
    }

    DIM = get_int("BasicSetting.DIM", 3);
    printf("Perform calculation in %d dimension\n", DIM);

    Particle* P;
    int npart = Read_Particle_File(&P);

#ifdef USE_OPENCL
    Setup_OpenCL("src/force_gpu.cl", npart);
    float eps2 = 0.01f;
#endif

    double T_TOT = get_double("BasicSetting.T_TOT", 0.0);
    double DT = get_double("BasicSetting.DT", 0.1);
    double t = 0.0;
    int STEP_PER_OUT = get_int("BasicSetting.STEP_PER_OUT", -1);
    double TIME_PER_OUT = get_double("BasicSetting.TIME_PER_OUT", -0.1);
    int step = 0;
    int time_step = 0;
    double dt = 0.0;

    while (t < T_TOT) {
        struct timeval t0, t1;
        gettimeofday(&t0, 0);
        step++;

#ifndef USE_OPENCL
        dt = Evolution(P, npart, Min(DT, T_TOT - t), dt);
#else
        Compute_Force_OpenCL(P, npart, eps2);
        dt = Evolution(P, npart, Min(DT, T_TOT - t), dt);
#endif

        t = Min(t + dt + 1e-15, T_TOT);

        gettimeofday(&t1, 0);
#ifdef DEBUG
        printf("Step %5d: x = (%lf, %lf, %lf)\n", step, P[0].x[0], P[0].x[1], P[0].x[2]);
        printf("Step %5d: f = (%lf, %lf, %lf)\n", step, P[0].f[0], P[0].f[1], P[0].f[2]);
#endif
        printf("Step %5d: t = %lf,  dt = %lf  timeElapsed: %lu ms\n", step, t, dt, (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000);

        if (((STEP_PER_OUT == -1) && (t / TIME_PER_OUT >= (double)time_step + 1.0)) ||
            ((STEP_PER_OUT != -1) && (step % STEP_PER_OUT == 0))) {
            time_step++;
            char filename[32];
            sprintf(filename, "%s/%05d.dat", OUTDIR, time_step);
            Write_Particle_File(P, npart, filename);
            printf("Data written to %s (t = %.3f)\n", filename, t);
        }
    }

    Write_Particle_File(P, npart, "Final.dat");
    printf("Data written to Final.dat\n");

#ifdef USE_OPENCL
    Cleanup_OpenCL();
#endif

    free(P);
    return 0;
}
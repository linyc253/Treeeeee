// #include <stdio.h>
// #include <stdlib.h>
// #include <OpenCL/opencl.h>

// typedef struct {
//     float x, y, z, w;
// } float4;

// #define N 1024
// #define EPS2 0.01f

// int main() {
//     float4 *h_pos = malloc(sizeof(float4) * N);
//     float4 *h_acc = malloc(sizeof(float4) * N);

//     // Sample initialization
//     for (int i = 0; i < N; i++) {
//         h_pos[i].x = i * 0.01f;
//         h_pos[i].y = 0.0f;
//         h_pos[i].z = 0.0f;
//         h_pos[i].w = 1.0f; // mass
//     }

//     // Load kernel source
//     FILE *fp = fopen("src/force_gpu.cl", "r");
//     fseek(fp, 0, SEEK_END);
//     size_t size = ftell(fp);
//     rewind(fp);
//     char *source = malloc(size + 1);
//     fread(source, 1, size, fp);
//     source[size] = '\0';
//     fclose(fp);

//     cl_int err;
//     cl_platform_id platform;
//     cl_device_id device;
//     cl_context context;
//     cl_command_queue queue;

//     clGetPlatformIDs(1, &platform, NULL);
//     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//     context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
//     queue = clCreateCommandQueue(context, device, 0, &err);

//     cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
//     clBuildProgram(program, 1, &device, NULL, NULL, NULL);
//     cl_kernel kernel = clCreateKernel(program, "compute_force", &err);

//     cl_mem d_pos = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float4) * N, h_pos, &err);
//     cl_mem d_acc = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float4) * N, NULL, &err);

//     int Nval = N;
//     float EPS2val = EPS2;

//     clSetKernelArg(kernel, 0, sizeof(int), &Nval);
//     clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_pos);
//     clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_acc);
//     clSetKernelArg(kernel, 3, sizeof(float), &EPS2val);

//     size_t global = N;
//     clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
//     clEnqueueReadBuffer(queue, d_acc, CL_TRUE, 0, sizeof(float4) * N, h_acc, 0, NULL, NULL);

//     printf("First acceleration result: (%f, %f, %f)\n", h_acc[0].x, h_acc[0].y, h_acc[0].z);

//     clReleaseMemObject(d_pos);
//     clReleaseMemObject(d_acc);
//     clReleaseKernel(kernel);
//     clReleaseProgram(program);
//     clReleaseCommandQueue(queue);
//     clReleaseContext(context);
//     free(h_pos);
//     free(h_acc);
//     free(source);
//     return 0;
// }

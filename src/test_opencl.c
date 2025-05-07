#include <stdio.h>
#include <OpenCL/opencl.h>

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting platform\n");
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting device\n");
        return 1;
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    printf("OpenCL setup complete! Running on Apple GPU.\n");

    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

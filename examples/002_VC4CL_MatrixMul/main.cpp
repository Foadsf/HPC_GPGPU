/**
 * VC4CL OpenCL Matrix Multiplication for Raspberry Pi 3B
 * Tiled Optimization Version
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>

// ============================================================================
// Configuration
// ============================================================================

static int MATRIX_DIM = 64;
static int NUM_ITERATIONS = 10;
// Work Group Size (Must be <= 12 on Pi 3B for some kernels)
#define BLOCK_SIZE 8 

// ============================================================================
// Utility Functions
// ============================================================================

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static char* load_kernel_source(const char* filename, size_t* length) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open kernel file '%s': %s\n", 
                filename, strerror(errno));
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* source = (char*)malloc(*length + 1);
    if (!source) {
        fclose(f);
        return NULL;
    }
    
    size_t read_size = fread(source, 1, *length, f);
    source[read_size] = '\0';
    *length = read_size;
    fclose(f);
    
    return source;
}

static const char* cl_error_string(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        default: return "Unknown OpenCL error";
    }
}

static void cpu_matrix_multiply(const float* A, const float* B, float* C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char* argv[]) {
    cl_int err;
    int ret = 0;
    
    if (argc >= 2) {
        MATRIX_DIM = atoi(argv[1]);
        // Tiling requires dimensions to be multiples of 16 and 8
        if (MATRIX_DIM < 16 || MATRIX_DIM > 1024 || MATRIX_DIM % 16 != 0) {
            fprintf(stderr, "Matrix dimension must be between 16 and 1024, and multiple of 16\n");
            return 1;
        }
    }
    if (argc >= 3) {
        NUM_ITERATIONS = atoi(argv[2]);
    }
    
    const int dim = MATRIX_DIM;
    const int size = dim * dim;
    const size_t bytes = size * sizeof(float);
    
    printf("=========================================\n");
    printf(" VC4CL OpenCL Matrix Multiplication\n");
    printf(" Target: Raspberry Pi 3B (VideoCore IV)\n");
    printf(" Mode:   Tiled (Local Memory)\n");
    printf("=========================================\n\n");
    
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem buf_A = NULL;
    cl_mem buf_B = NULL;
    cl_mem buf_C = NULL;
    float* A = NULL;
    float* B = NULL;
    float* C_cpu = NULL;
    float* C_gpu = NULL;
    char* source = NULL;

    // Benchmark variables declared top-level to avoid goto errors
    double cpu_total = 0.0;
    double cpu_avg = 0.0;
    double cpu_gflops = 0.0;
    double gpu_total = 0.0;
    double gpu_avg = 0.0;
    double gpu_gflops = 0.0;
    
    // Moved these to top level
    int cpu_iters = 0;
    int err_count = 0;

    // WORK GROUP SETUP
    // Global: Divide width by 16 (vectorized)
    // Local:  Use 8 threads per group (must be <= 12)
    size_t global_work_size[2] = { (size_t)dim / 16, (size_t)dim };
    size_t local_work_size[2]  = { BLOCK_SIZE, 1 }; 

    // Validate size
    if (global_work_size[0] % local_work_size[0] != 0) {
        fprintf(stderr, "Error: Global work size X (%zu) not divisible by Local size X (%d)\n",
                global_work_size[0], BLOCK_SIZE);
        return 1;
    }
    
    // Platform/Device boilerplate
    cl_platform_id platform;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platform, &num_platforms);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    // Print constraints
    size_t max_wg;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    printf("Hardware Max Work Group Size: %zu\n", max_wg);
    printf("Selected Work Group Size: %d\n", BLOCK_SIZE);
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err) goto cleanup;
    
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err) goto cleanup;
    
    // Build Program
    size_t source_len;
    source = load_kernel_source("matmul.cl", &source_len);
    if (!source) goto cleanup;
    
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_len, &err);
    
    // Build with optimization and define BLOCK_SIZE for the kernel
    char build_opts[64];
    sprintf(build_opts, "-cl-fast-relaxed-math -DBLOCK_SIZE=%d", BLOCK_SIZE);
    err = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Build Error:\n%s\n", log);
        free(log);
        goto cleanup;
    }
    
    kernel = clCreateKernel(program, "matmul_tiled", &err);
    if (err) goto cleanup;
    
    // Alloc Host
    A = (float*)malloc(bytes);
    B = (float*)malloc(bytes);
    C_cpu = (float*)malloc(bytes);
    C_gpu = (float*)malloc(bytes);
    
    srand(42);
    for (int i=0; i<size; i++) {
        A[i] = (float)rand()/RAND_MAX;
        B[i] = (float)rand()/RAND_MAX;
    }
    
    // CPU Run
    printf("--- CPU Benchmark ---\n");
    // Reduce iterations for CPU if large matrix
    cpu_iters = (dim > 256) ? 1 : NUM_ITERATIONS;
    
    for(int i=0; i<cpu_iters; i++) {
        double t1 = get_time_ms();
        cpu_matrix_multiply(A, B, C_cpu, dim);
        cpu_total += (get_time_ms() - t1);
    }
    cpu_avg = cpu_total / cpu_iters;
    cpu_gflops = (2.0 * dim * dim * dim) / (cpu_avg * 1e6);
    printf("CPU Avg: %.2f ms | %.3f GFLOPS\n", cpu_avg, cpu_gflops);
    
    // GPU Setup
    buf_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A, NULL);
    buf_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B, NULL);
    buf_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_C);
    clSetKernelArg(kernel, 3, sizeof(int), &dim);
    
    printf("\n--- GPU Benchmark (Tiled) ---\n");
    printf("Work Size: Global[%zu, %zu] Local[%zu, %zu]\n", 
           global_work_size[0], global_work_size[1], local_work_size[0], local_work_size[1]);

    // Warmup
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    if (err) {
        fprintf(stderr, "Kernel Launch Error: %s\n", cl_error_string(err));
        goto cleanup;
    }
    clFinish(queue);
    
    for(int i=0; i<NUM_ITERATIONS; i++) {
        double t1 = get_time_ms();
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        clFinish(queue);
        gpu_total += (get_time_ms() - t1);
    }
    
    gpu_avg = gpu_total / NUM_ITERATIONS;
    gpu_gflops = (2.0 * dim * dim * dim) / (gpu_avg * 1e6);
    
    clEnqueueReadBuffer(queue, buf_C, CL_TRUE, 0, bytes, C_gpu, 0, NULL, NULL);
    
    // Verify
    err_count = 0;
    for(int i=0; i<size; i++) {
        float diff = fabs(C_cpu[i] - C_gpu[i]);
        if (diff > 0.01) err_count++; // Looser tolerance for fast-math
    }
    
    printf("\nResults:\n");
    printf("GPU Avg: %.2f ms\n", gpu_avg);
    printf("GPU Perf: %.3f GFLOPS\n", gpu_gflops);
    printf("Speedup: %.2fx\n", cpu_avg / gpu_avg);
    printf("Errors > 0.01: %d/%d\n", err_count, size);

cleanup:
    if(buf_A) clReleaseMemObject(buf_A);
    if(buf_B) clReleaseMemObject(buf_B);
    if(buf_C) clReleaseMemObject(buf_C);
    if(kernel) clReleaseKernel(kernel);
    if(program) clReleaseProgram(program);
    if(queue) clReleaseCommandQueue(queue);
    if(context) clReleaseContext(context);
    free(source);
    free(A); free(B); free(C_cpu); free(C_gpu);
    return ret;
}

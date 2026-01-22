/**
 * VC4CL OpenCL Matrix Multiplication for Raspberry Pi 3B
 * * Uses the VC4CL OpenCL implementation for VideoCore IV QPUs
 * * Build: cmake .. && make
 * Run:   ./vc4cl_mm [matrix_size] [iterations]
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

// ============================================================================
// CPU Reference Implementation
// ============================================================================

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
    
    // Parse command line arguments
    if (argc >= 2) {
       MATRIX_DIM = atoi(argv[1]);
       // Increased limit to 1024 to test GPU scaling
       if (MATRIX_DIM < 8 || MATRIX_DIM > 1024) {
           fprintf(stderr, "Matrix dimension must be between 8 and 1024\n");
           return 1;
       }
   }
    if (argc >= 3) {
        NUM_ITERATIONS = atoi(argv[2]);
        if (NUM_ITERATIONS < 1 || NUM_ITERATIONS > 100) {
            fprintf(stderr, "Iterations must be between 1 and 100\n");
            return 1;
        }
    }
    
    const int dim = MATRIX_DIM;
    const int size = dim * dim;
    const size_t bytes = size * sizeof(float);
    
    printf("=========================================\n");
    printf(" VC4CL OpenCL Matrix Multiplication\n");
    printf(" Target: Raspberry Pi 3B (VideoCore IV)\n");
    printf("=========================================\n\n");
    
    // Initialize all pointers to NULL for safe cleanup
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

    // --- FIX: Declare variables here to avoid jump errors ---
    double cpu_total = 0.0;
    double cpu_avg = 0.0;
    double cpu_gflops = 0.0;
    
    double gpu_total = 0.0;
    double gpu_avg = 0.0;
    double gpu_gflops = 0.0;
    
    // VECTORIZATION: Launch 16x fewer threads in the X dimension
    // Each thread will compute 16 elements at once.
    size_t global_work_size[2] = { (size_t)dim / 16, (size_t)dim };
    // --------------------------------------------------------
    
    // ========================================================================
    // OpenCL Platform/Device Setup
    // ========================================================================
    
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error: No OpenCL platforms found (%s)\n", cl_error_string(err));
        return 1;
    }
    
    char platform_name[256];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    printf("Platform: %s\n", platform_name);
    
    cl_device_id device;
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        fprintf(stderr, "Error: No GPU devices found (%s)\n", cl_error_string(err));
        return 1;
    }
    
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Device: %s\n", device_name);
    
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    printf("Max work group size: %zu\n", max_work_group_size);
    
    cl_uint max_compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
    printf("Max compute units: %u\n", max_compute_units);
    
    printf("\nMatrix size: %dx%d (%d elements)\n", dim, dim, size);
    printf("Iterations: %d\n", NUM_ITERATIONS);
    printf("FLOPs per matmul: %lld (2*N^3)\n\n", 2LL * dim * dim * dim);
    
    // ========================================================================
    // Create OpenCL Context and Command Queue
    // ========================================================================
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create context (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create command queue (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    // ========================================================================
    // Load and Build Kernel
    // ========================================================================
    
    printf("--- Building Kernel ---\n");
    
    size_t source_length;
    source = load_kernel_source("matmul.cl", &source_length);
    if (!source) {
        ret = 1;
        goto cleanup;
    }
    
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_length, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create program (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    // Build with optimization flags for VC4CL
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to build program (%s)\n", cl_error_string(err));
        
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "Build log:\n%s\n", log);
            free(log);
        }
        
        ret = 1;
        goto cleanup;
    }
    printf("Kernel compiled successfully\n\n");
    
    kernel = clCreateKernel(program, "matmul_simple", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create kernel (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    // ========================================================================
    // Allocate Host Memory
    // ========================================================================
    
    A = (float*)malloc(bytes);
    B = (float*)malloc(bytes);
    C_cpu = (float*)malloc(bytes);
    C_gpu = (float*)malloc(bytes);
    
    if (!A || !B || !C_cpu || !C_gpu) {
        fprintf(stderr, "Error: Failed to allocate host memory\n");
        ret = 1;
        goto cleanup;
    }
    
    // Initialize matrices
    srand(42);
    for (int i = 0; i < size; i++) {
        A[i] = (float)rand() / (float)RAND_MAX;
        B[i] = (float)rand() / (float)RAND_MAX;
    }
    
    // ========================================================================
    // CPU Benchmark
    // ========================================================================
    
    printf("--- CPU Benchmark ---\n");
    
    // cpu_total declared at top
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double start = get_time_ms();
        cpu_matrix_multiply(A, B, C_cpu, dim);
        double end = get_time_ms();
        cpu_total += (end - start);
    }
    cpu_avg = cpu_total / NUM_ITERATIONS;
    cpu_gflops = (2.0 * dim * dim * dim) / (cpu_avg * 1e6);
    
    printf("CPU Total Time: %.2f ms (%d iterations)\n", cpu_total, NUM_ITERATIONS);
    printf("CPU Avg Time: %.2f ms per matmul\n", cpu_avg);
    printf("CPU Performance: %.3f GFLOPS\n\n", cpu_gflops);
    
    // ========================================================================
    // Create OpenCL Buffers
    // ========================================================================
    
    buf_A = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create buffer A (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    buf_B = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create buffer B (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    buf_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create buffer C (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    // Upload data
    err = clEnqueueWriteBuffer(queue, buf_A, CL_TRUE, 0, bytes, A, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to upload A (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    err = clEnqueueWriteBuffer(queue, buf_B, CL_TRUE, 0, bytes, B, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to upload B (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &dim);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to set kernel arguments (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    // ========================================================================
    // GPU Benchmark
    // ========================================================================
    
    printf("--- GPU (VC4CL) Benchmark ---\n");
    
    // global_work_size declared at top
    
    // Warm-up
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Kernel execution failed (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    // Timed runs
    // gpu_total declared at top
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double start = get_time_ms();
        
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
        clFinish(queue);
        
        double end = get_time_ms();
        gpu_total += (end - start);
        
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error: Kernel execution failed on iteration %d (%s)\n", iter, cl_error_string(err));
            ret = 1;
            goto cleanup;
        }
    }
    
    gpu_avg = gpu_total / NUM_ITERATIONS;
    gpu_gflops = (2.0 * dim * dim * dim) / (gpu_avg * 1e6);
    
    printf("GPU Total Time: %.2f ms (%d iterations)\n", gpu_total, NUM_ITERATIONS);
    printf("GPU Avg Time: %.2f ms per matmul\n", gpu_avg);
    printf("GPU Performance: %.3f GFLOPS\n\n", gpu_gflops);
    
    // Read back results
    err = clEnqueueReadBuffer(queue, buf_C, CL_TRUE, 0, bytes, C_gpu, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to read results (%s)\n", cl_error_string(err));
        ret = 1;
        goto cleanup;
    }
    
    // ========================================================================
    // Verify Results
    // ========================================================================
    
    {
        double max_error = 0.0;
        double avg_error = 0.0;
        int error_count = 0;
        
        for (int i = 0; i < size; i++) {
            double error = fabs((double)C_cpu[i] - (double)C_gpu[i]);
            avg_error += error;
            if (error > max_error) max_error = error;
            if (error > 0.001) {
                error_count++;
            }
        }
        avg_error /= size;
        
        // ========================================================================
        // Summary
        // ========================================================================
        
        printf("=========================================\n");
        printf("RESULTS SUMMARY\n");
        printf("=========================================\n");
        printf("Matrix Size: %d x %d\n", dim, dim);
        printf("Iterations: %d\n\n", NUM_ITERATIONS);
        
        printf("Timing (avg per matmul):\n");
        printf("  CPU: %.2f ms\n", cpu_avg);
        printf("  GPU: %.2f ms\n", gpu_avg);
        printf("  Speedup: %.2fx\n\n", cpu_avg / gpu_avg);
        
        printf("Performance:\n");
        printf("  CPU: %.3f GFLOPS\n", cpu_gflops);
        printf("  GPU: %.3f GFLOPS\n", gpu_gflops);
        printf("  GPU Theoretical Peak: ~24 GFLOPS (12 QPUs)\n\n");
        
        printf("Accuracy:\n");
        printf("  Max Error: %.6f\n", max_error);
        printf("  Avg Error: %.6f\n", avg_error);
        printf("  Errors > 0.001: %d / %d\n", error_count, size);
        printf("=========================================\n");
    }
    
cleanup:
    // Free OpenCL resources
    if (buf_A) clReleaseMemObject(buf_A);
    if (buf_B) clReleaseMemObject(buf_B);
    if (buf_C) clReleaseMemObject(buf_C);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    
    // Free host memory
    free(source);
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    
    return ret;
}

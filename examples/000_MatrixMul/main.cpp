/**
 * OpenGL ES 2.0 GPGPU Matrix Multiplication for Raspberry Pi 3B
 * OPTIMIZED VERSION - Demonstrates actual GPU speedup
 * 
 * Key optimizations:
 * 1. Larger matrices (128x128) - more work to amortize setup cost
 * 2. Multiple iterations - benchmark just the kernel execution
 * 3. Separate timing for setup vs compute vs readback
 * 4. Warm-up run to prime caches and JIT
 * 
 * Build: cmake .. && make
 * Run:   ./gpgpu_mm [matrix_size] [iterations]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>

// EGL and OpenGL ES 2.0 headers
#include <EGL/egl.h>
#include <GLES2/gl2.h>

// For headless rendering on modern Pi OS (DRM/GBM)
#include <fcntl.h>
#include <unistd.h>
#include <gbm.h>

// ============================================================================
// Configuration (can be overridden via command line)
// ============================================================================

static int MATRIX_DIM = 128;    // Default: 128x128 (larger = better GPU utilization)
static int NUM_ITERATIONS = 10; // Run multiple times to get stable timing

// ============================================================================
// Utility Functions
// ============================================================================

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static char* load_shader_source(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open shader file '%s': %s\n", 
                filename, strerror(errno));
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* source = (char*)malloc(size + 1);
    if (!source) {
        fclose(f);
        return NULL;
    }
    
    size_t read_size = fread(source, 1, size, f);
    source[read_size] = '\0';
    fclose(f);
    
    return source;
}

static GLuint compile_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "Shader compilation failed:\n%s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

static GLuint create_program(const char* vs_source, const char* fs_source) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
    if (!vs) return 0;
    
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source);
    if (!fs) {
        glDeleteShader(vs);
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    
    glDeleteShader(vs);
    glDeleteShader(fs);
    
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "Program linking failed:\n%s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    
    return program;
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
// EGL Setup for Headless Rendering (GBM backend)
// ============================================================================

static int drm_fd = -1;
static struct gbm_device* gbm_dev = NULL;
static struct gbm_surface* gbm_surf = NULL;
static EGLDisplay egl_display = EGL_NO_DISPLAY;
static EGLContext egl_context = EGL_NO_CONTEXT;
static EGLSurface egl_surface = EGL_NO_SURFACE;

static int init_egl_gbm(void) {
    const char* card = "/dev/dri/card0";
    drm_fd = open(card, O_RDWR);
    if (drm_fd < 0) {
        card = "/dev/dri/card1";
        drm_fd = open(card, O_RDWR);
    }
    if (drm_fd < 0) {
        fprintf(stderr, "Error: Cannot open DRM device\n");
        return -1;
    }
    
    gbm_dev = gbm_create_device(drm_fd);
    if (!gbm_dev) {
        fprintf(stderr, "Error: Cannot create GBM device\n");
        close(drm_fd);
        return -1;
    }
    
    egl_display = eglGetDisplay((EGLNativeDisplayType)gbm_dev);
    if (egl_display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Error: Cannot get EGL display\n");
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    EGLint major, minor;
    if (!eglInitialize(egl_display, &major, &minor)) {
        fprintf(stderr, "Error: Cannot initialize EGL\n");
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        fprintf(stderr, "Error: Cannot bind OpenGL ES API\n");
        eglTerminate(egl_display);
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    static const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    
    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(egl_display, config_attribs, &config, 1, &num_configs) 
        || num_configs == 0) {
        fprintf(stderr, "Error: Cannot choose EGL config\n");
        eglTerminate(egl_display);
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    gbm_surf = gbm_surface_create(gbm_dev, 256, 256,
                                   GBM_FORMAT_ARGB8888,
                                   GBM_BO_USE_RENDERING);
    if (!gbm_surf) {
        fprintf(stderr, "Error: Cannot create GBM surface\n");
        eglTerminate(egl_display);
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    egl_surface = eglCreateWindowSurface(egl_display, config, 
                                          (EGLNativeWindowType)gbm_surf, NULL);
    if (egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "Error: Cannot create EGL surface\n");
        gbm_surface_destroy(gbm_surf);
        eglTerminate(egl_display);
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    static const EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    
    egl_context = eglCreateContext(egl_display, config, EGL_NO_CONTEXT, 
                                    context_attribs);
    if (egl_context == EGL_NO_CONTEXT) {
        fprintf(stderr, "Error: Cannot create EGL context\n");
        eglDestroySurface(egl_display, egl_surface);
        gbm_surface_destroy(gbm_surf);
        eglTerminate(egl_display);
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    if (!eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) {
        fprintf(stderr, "Error: Cannot make EGL context current\n");
        eglDestroyContext(egl_display, egl_context);
        eglDestroySurface(egl_display, egl_surface);
        gbm_surface_destroy(gbm_surf);
        eglTerminate(egl_display);
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    return 0;
}

static void cleanup_egl(void) {
    if (egl_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (egl_context != EGL_NO_CONTEXT)
            eglDestroyContext(egl_display, egl_context);
        if (egl_surface != EGL_NO_SURFACE)
            eglDestroySurface(egl_display, egl_surface);
        eglTerminate(egl_display);
    }
    if (gbm_surf)
        gbm_surface_destroy(gbm_surf);
    if (gbm_dev)
        gbm_device_destroy(gbm_dev);
    if (drm_fd >= 0)
        close(drm_fd);
}

// ============================================================================
// Generate fragment shader with loop unrolled for specific matrix size
// ============================================================================

static char* generate_fragment_shader(int matrix_dim) {
    // For larger matrices, we need to handle the loop carefully
    // The VideoCore IV GLSL compiler works better with explicit loop bounds
    
    char* shader = (char*)malloc(4096);
    if (!shader) return NULL;
    
    snprintf(shader, 4096,
        "// Auto-generated fragment shader for %dx%d matrix multiplication\n"
        "precision mediump float;\n"
        "\n"
        "uniform sampler2D u_matrixA;\n"
        "uniform sampler2D u_matrixB;\n"
        "uniform float u_width;\n"
        "\n"
        "varying vec2 v_texcoord;\n"
        "\n"
        "void main() {\n"
        "    float row = floor(v_texcoord.y * u_width);\n"
        "    float col = floor(v_texcoord.x * u_width);\n"
        "    \n"
        "    float sum = 0.0;\n"
        "    float invWidth = 1.0 / u_width;\n"
        "    \n"
        "    // Loop with compile-time constant upper bound\n"
        "    for (float k = 0.0; k < %.1f; k += 1.0) {\n"
        "        vec2 coordA = vec2((k + 0.5) * invWidth, (row + 0.5) * invWidth);\n"
        "        vec2 coordB = vec2((col + 0.5) * invWidth, (k + 0.5) * invWidth);\n"
        "        sum += texture2D(u_matrixA, coordA).r * texture2D(u_matrixB, coordB).r;\n"
        "    }\n"
        "    \n"
        "    // Normalize result to [0,1] range\n"
        "    float result = sum / u_width;\n"
        "    gl_FragColor = vec4(result, result, result, 1.0);\n"
        "}\n",
        matrix_dim, matrix_dim, (float)matrix_dim
    );
    
    return shader;
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc >= 2) {
        MATRIX_DIM = atoi(argv[1]);
        if (MATRIX_DIM < 8 || MATRIX_DIM > 512) {
            fprintf(stderr, "Matrix dimension must be between 8 and 512\n");
            return 1;
        }
    }
    if (argc >= 3) {
        NUM_ITERATIONS = atoi(argv[2]);
        if (NUM_ITERATIONS < 1 || NUM_ITERATIONS > 1000) {
            fprintf(stderr, "Iterations must be between 1 and 1000\n");
            return 1;
        }
    }
    
    const int dim = MATRIX_DIM;
    const int size = dim * dim;
    
    printf("=========================================\n");
    printf(" OpenGL ES 2.0 GPGPU Matrix Multiplication\n");
    printf(" Target: Raspberry Pi 3B (VideoCore IV)\n");
    printf(" OPTIMIZED VERSION\n");
    printf("=========================================\n\n");
    
    // Initialize EGL
    if (init_egl_gbm() != 0) {
        fprintf(stderr, "Failed to initialize EGL\n");
        return 1;
    }
    
    // Print GPU info
    printf("GPU: %s\n", glGetString(GL_RENDERER));
    printf("OpenGL ES: %s\n", glGetString(GL_VERSION));
    printf("\nMatrix size: %dx%d (%d elements)\n", dim, dim, size);
    printf("Iterations: %d\n", NUM_ITERATIONS);
    printf("FLOPs per matmul: %lld (2*N^3)\n\n", 2LL * dim * dim * dim);
    
    // Allocate matrices
    float* A_float = (float*)malloc(size * sizeof(float));
    float* B_float = (float*)malloc(size * sizeof(float));
    float* C_cpu = (float*)malloc(size * sizeof(float));
    float* C_gpu = (float*)malloc(size * sizeof(float));
    
    if (!A_float || !B_float || !C_cpu || !C_gpu) {
        fprintf(stderr, "Failed to allocate matrices\n");
        cleanup_egl();
        return 1;
    }
    
    // Initialize with random values in [0, 1] range
    srand(42);
    for (int i = 0; i < size; i++) {
        A_float[i] = (float)rand() / (float)RAND_MAX;
        B_float[i] = (float)rand() / (float)RAND_MAX;
    }
    
    // ========================================================================
    // CPU Benchmark
    // ========================================================================
    printf("--- CPU Benchmark ---\n");
    
    double cpu_total = 0.0;
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double start = get_time_ms();
        cpu_matrix_multiply(A_float, B_float, C_cpu, dim);
        double end = get_time_ms();
        cpu_total += (end - start);
    }
    double cpu_avg = cpu_total / NUM_ITERATIONS;
    double cpu_gflops = (2.0 * dim * dim * dim) / (cpu_avg * 1e6);
    
    printf("CPU Total Time: %.2f ms (%d iterations)\n", cpu_total, NUM_ITERATIONS);
    printf("CPU Avg Time: %.2f ms per matmul\n", cpu_avg);
    printf("CPU Performance: %.3f GFLOPS\n\n", cpu_gflops);
    
    // ========================================================================
    // GPU Setup
    // ========================================================================
    printf("--- GPU Setup ---\n");
    
    double setup_start = get_time_ms();
    
    // Load vertex shader
    char* vs_source = load_shader_source("vertex.glsl");
    if (!vs_source) {
        cleanup_egl();
        return 1;
    }
    
    // Generate fragment shader for this matrix size
    char* fs_source = generate_fragment_shader(dim);
    if (!fs_source) {
        free(vs_source);
        cleanup_egl();
        return 1;
    }
    
    GLuint program = create_program(vs_source, fs_source);
    free(vs_source);
    free(fs_source);
    if (!program) {
        cleanup_egl();
        return 1;
    }
    
    // Get uniform/attribute locations
    GLint a_position = glGetAttribLocation(program, "a_position");
    GLint a_texcoord = glGetAttribLocation(program, "a_texcoord");
    GLint u_matrixA = glGetUniformLocation(program, "u_matrixA");
    GLint u_matrixB = glGetUniformLocation(program, "u_matrixB");
    GLint u_width = glGetUniformLocation(program, "u_width");
    
    // Create full-screen quad
    static const float quad_vertices[] = {
        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
        -1.0f,  1.0f,   0.0f, 1.0f,
         1.0f,  1.0f,   1.0f, 1.0f,
    };
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
    
    // Convert matrices to RGBA8 textures
    unsigned char* texA = (unsigned char*)calloc(size * 4, 1);
    unsigned char* texB = (unsigned char*)calloc(size * 4, 1);
    
    for (int i = 0; i < size; i++) {
        unsigned char valA = (unsigned char)(A_float[i] * 255.0f);
        unsigned char valB = (unsigned char)(B_float[i] * 255.0f);
        texA[i * 4 + 0] = valA;
        texA[i * 4 + 1] = valA;
        texA[i * 4 + 2] = valA;
        texA[i * 4 + 3] = 255;
        texB[i * 4 + 0] = valB;
        texB[i * 4 + 1] = valB;
        texB[i * 4 + 2] = valB;
        texB[i * 4 + 3] = 255;
    }
    
    // Create input textures
    GLuint tex_A, tex_B;
    glGenTextures(1, &tex_A);
    glBindTexture(GL_TEXTURE_2D, tex_A);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, dim, dim, 0, GL_RGBA, GL_UNSIGNED_BYTE, texA);
    
    glGenTextures(1, &tex_B);
    glBindTexture(GL_TEXTURE_2D, tex_B);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, dim, dim, 0, GL_RGBA, GL_UNSIGNED_BYTE, texB);
    
    free(texA);
    free(texB);
    
    // Create output texture (render target)
    GLuint tex_C;
    glGenTextures(1, &tex_C);
    glBindTexture(GL_TEXTURE_2D, tex_C);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, dim, dim, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    
    // Create FBO
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_C, 0);
    
    GLenum fbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fbo_status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Error: FBO incomplete (status: 0x%x)\n", fbo_status);
        cleanup_egl();
        return 1;
    }
    
    // Set up render state (once)
    glViewport(0, 0, dim, dim);
    glUseProgram(program);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_A);
    glUniform1i(u_matrixA, 0);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_B);
    glUniform1i(u_matrixB, 1);
    
    glUniform1f(u_width, (float)dim);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(a_position);
    glVertexAttribPointer(a_position, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(a_texcoord);
    glVertexAttribPointer(a_texcoord, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 
                          (void*)(2 * sizeof(float)));
    
    double setup_end = get_time_ms();
    printf("GPU Setup Time: %.2f ms\n\n", setup_end - setup_start);
    
    // ========================================================================
    // GPU Benchmark (warm-up + timed runs)
    // ========================================================================
    printf("--- GPU Benchmark ---\n");
    
    // Warm-up run (primes caches, JIT, etc.)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFinish();
    
    // Timed runs
    double gpu_total = 0.0;
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        double start = get_time_ms();
        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glFinish();  // Ensure GPU is done
        
        double end = get_time_ms();
        gpu_total += (end - start);
    }
    double gpu_avg = gpu_total / NUM_ITERATIONS;
    double gpu_gflops = (2.0 * dim * dim * dim) / (gpu_avg * 1e6);
    
    printf("GPU Compute Time: %.2f ms (%d iterations)\n", gpu_total, NUM_ITERATIONS);
    printf("GPU Avg Time: %.2f ms per matmul\n", gpu_avg);
    printf("GPU Performance: %.3f GFLOPS\n\n", gpu_gflops);
    
    // ========================================================================
    // Read back results
    // ========================================================================
    printf("--- Readback ---\n");
    double readback_start = get_time_ms();
    
    unsigned char* result_rgba = (unsigned char*)malloc(size * 4);
    glReadPixels(0, 0, dim, dim, GL_RGBA, GL_UNSIGNED_BYTE, result_rgba);
    
    for (int i = 0; i < size; i++) {
        float normalized = result_rgba[i * 4] / 255.0f;
        C_gpu[i] = normalized * dim;
    }
    
    free(result_rgba);
    
    double readback_end = get_time_ms();
    printf("Readback Time: %.2f ms\n\n", readback_end - readback_start);
    
    // ========================================================================
    // Verify results
    // ========================================================================
    double max_error = 0.0;
    double avg_error = 0.0;
    int error_count = 0;
    
    for (int i = 0; i < size; i++) {
        double error = fabs(C_cpu[i] - C_gpu[i]);
        avg_error += error;
        if (error > max_error) max_error = error;
        if (error > C_cpu[i] * 0.1 + 0.5) {
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
    printf("  GPU: %.2f ms (compute only)\n", gpu_avg);
    printf("  Speedup: %.2fx\n\n", cpu_avg / gpu_avg);
    
    printf("Performance:\n");
    printf("  CPU: %.3f GFLOPS\n", cpu_gflops);
    printf("  GPU: %.3f GFLOPS\n", gpu_gflops);
    printf("  GPU Theoretical Peak: ~24 GFLOPS (VideoCore IV)\n\n");
    
    printf("Accuracy:\n");
    printf("  Max Error: %.6f\n", max_error);
    printf("  Avg Error: %.6f\n", avg_error);
    printf("  Large Errors: %d / %d (%.2f%%)\n", 
           error_count, size, 100.0 * error_count / size);
    printf("  Note: Error due to 8-bit quantization (~0.4%% precision)\n");
    printf("=========================================\n");
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex_A);
    glDeleteTextures(1, &tex_B);
    glDeleteTextures(1, &tex_C);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    
    free(A_float);
    free(B_float);
    free(C_cpu);
    free(C_gpu);
    
    cleanup_egl();
    
    return 0;
}

/**
 * OpenGL ES 2.0 GPGPU Matrix Multiplication for Raspberry Pi 3B
 * 
 * This demonstrates actual GPU-accelerated computation on the VideoCore IV
 * using fragment shader GPGPU with render-to-texture via FBOs.
 * 
 * Key technique: Since VideoCore IV lacks compute shaders and float textures,
 * we use the classic "render-to-texture" GPGPU approach with RGBA8 textures
 * and encode/decode our data appropriately.
 * 
 * Build: cmake .. && make
 * Run:   ./gpgpu_mm
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
// Configuration
// ============================================================================

// Matrix dimension (NxN square matrices)
// Keep small due to per-fragment texture fetch limits on VideoCore IV
// The fragment shader loop limit and texture fetch overhead constrains this
#define MATRIX_DIM 64
#define MATRIX_SIZE (MATRIX_DIM * MATRIX_DIM)

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
    
    size_t read = fread(source, 1, size, f);
    source[read] = '\0';
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
    
    // Shaders can be deleted after linking
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
    // Open DRM device (GPU)
    const char* card = "/dev/dri/card0";
    drm_fd = open(card, O_RDWR);
    if (drm_fd < 0) {
        // Try card1 (on some systems the GPU is card1)
        card = "/dev/dri/card1";
        drm_fd = open(card, O_RDWR);
    }
    if (drm_fd < 0) {
        fprintf(stderr, "Error: Cannot open DRM device\n");
        return -1;
    }
    printf("Opened DRM device: %s\n", card);
    
    // Create GBM device
    gbm_dev = gbm_create_device(drm_fd);
    if (!gbm_dev) {
        fprintf(stderr, "Error: Cannot create GBM device\n");
        close(drm_fd);
        return -1;
    }
    
    // Get EGL display from GBM device
    egl_display = eglGetDisplay((EGLNativeDisplayType)gbm_dev);
    if (egl_display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Error: Cannot get EGL display\n");
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    // Initialize EGL
    EGLint major, minor;
    if (!eglInitialize(egl_display, &major, &minor)) {
        fprintf(stderr, "Error: Cannot initialize EGL\n");
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    printf("EGL Version: %d.%d\n", major, minor);
    
    // Bind OpenGL ES API
    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        fprintf(stderr, "Error: Cannot bind OpenGL ES API\n");
        eglTerminate(egl_display);
        gbm_device_destroy(gbm_dev);
        close(drm_fd);
        return -1;
    }
    
    // Choose EGL config
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
    
    // Create GBM surface (needed for EGL surface)
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
    
    // Create EGL surface from GBM surface
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
    
    // Create OpenGL ES 2.0 context
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
    
    // Make context current
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
// Main Program
// ============================================================================

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    
    printf("=========================================\n");
    printf(" OpenGL ES 2.0 GPGPU Matrix Multiplication\n");
    printf(" Target: Raspberry Pi 3B (VideoCore IV)\n");
    printf("=========================================\n\n");
    
    // Initialize EGL with GBM backend (headless)
    if (init_egl_gbm() != 0) {
        fprintf(stderr, "Failed to initialize EGL\n");
        return 1;
    }
    
    // Print GPU info
    printf("=========================================\n");
    printf("  GPU: %s\n", glGetString(GL_RENDERER));
    printf("  Vendor: %s\n", glGetString(GL_VENDOR));
    printf("  OpenGL ES: %s\n", glGetString(GL_VERSION));
    printf("  GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("=========================================\n\n");
    
    // Check for useful extensions
    const char* extensions = (const char*)glGetString(GL_EXTENSIONS);
    printf("Float texture support: %s\n", 
           strstr(extensions, "OES_texture_float") ? "YES" : "NO (using fixed-point)");
    printf("\n");
    
    // Matrix configuration
    const int dim = MATRIX_DIM;
    const int size = MATRIX_SIZE;
    printf("Matrix size: %dx%d (%d elements)\n\n", dim, dim, size);
    
    // Allocate matrices
    float* A_float = (float*)malloc(size * sizeof(float));
    float* B_float = (float*)malloc(size * sizeof(float));
    float* C_cpu = (float*)malloc(size * sizeof(float));
    float* C_gpu = (float*)malloc(size * sizeof(float));
    
    // Initialize with random values in [0, 1] range
    // (Needed because we're using RGBA8 textures that only store [0,1])
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < size; i++) {
        A_float[i] = (float)rand() / (float)RAND_MAX;
        B_float[i] = (float)rand() / (float)RAND_MAX;
    }
    
    // ========================================================================
    // CPU Benchmark
    // ========================================================================
    printf("Starting CPU Matrix Multiplication...\n");
    double cpu_start = get_time_ms();
    cpu_matrix_multiply(A_float, B_float, C_cpu, dim);
    double cpu_end = get_time_ms();
    double cpu_time = cpu_end - cpu_start;
    printf("CPU Time: %.2f ms\n\n", cpu_time);
    
    // ========================================================================
    // GPU Setup
    // ========================================================================
    
    // Load shaders
    char* vs_source = load_shader_source("vertex.glsl");
    char* fs_source = load_shader_source("fragment.glsl");
    if (!vs_source || !fs_source) {
        fprintf(stderr, "Failed to load shaders\n");
        free(vs_source);
        free(fs_source);
        cleanup_egl();
        return 1;
    }
    
    GLuint program = create_program(vs_source, fs_source);
    free(vs_source);
    free(fs_source);
    if (!program) {
        fprintf(stderr, "Failed to create shader program\n");
        cleanup_egl();
        return 1;
    }
    
    // Get uniform/attribute locations
    GLint a_position = glGetAttribLocation(program, "a_position");
    GLint a_texcoord = glGetAttribLocation(program, "a_texcoord");
    GLint u_matrixA = glGetUniformLocation(program, "u_matrixA");
    GLint u_matrixB = glGetUniformLocation(program, "u_matrixB");
    GLint u_width = glGetUniformLocation(program, "u_width");
    
    // Create full-screen quad vertices
    // Position (x,y) and texture coord (s,t) interleaved
    static const float quad_vertices[] = {
        // Position     TexCoord
        -1.0f, -1.0f,   0.0f, 0.0f,  // Bottom-left
         1.0f, -1.0f,   1.0f, 0.0f,  // Bottom-right
        -1.0f,  1.0f,   0.0f, 1.0f,  // Top-left
         1.0f,  1.0f,   1.0f, 1.0f,  // Top-right
    };
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
    
    // Convert float matrices to RGBA8 texture data
    // Store value in R channel (values already in [0,1] range)
    unsigned char* texA = (unsigned char*)calloc(dim * dim * 4, 1);
    unsigned char* texB = (unsigned char*)calloc(dim * dim * 4, 1);
    
    for (int i = 0; i < size; i++) {
        // Clamp and convert to 8-bit
        unsigned char valA = (unsigned char)(A_float[i] * 255.0f);
        unsigned char valB = (unsigned char)(B_float[i] * 255.0f);
        
        texA[i * 4 + 0] = valA;  // R
        texA[i * 4 + 1] = valA;  // G (duplicate for easier debugging)
        texA[i * 4 + 2] = valA;  // B
        texA[i * 4 + 3] = 255;   // A
        
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
    
    // Create FBO and attach output texture
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
    
    // ========================================================================
    // GPU Benchmark
    // ========================================================================
    printf("Starting GPU Matrix Multiplication...\n");
    
    double gpu_start = get_time_ms();
    
    // Set viewport to match texture size
    glViewport(0, 0, dim, dim);
    
    // Use our shader program
    glUseProgram(program);
    
    // Bind input textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_A);
    glUniform1i(u_matrixA, 0);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_B);
    glUniform1i(u_matrixB, 1);
    
    // Set matrix dimension
    glUniform1f(u_width, (float)dim);
    
    // Setup vertex attributes
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(a_position);
    glVertexAttribPointer(a_position, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(a_texcoord);
    glVertexAttribPointer(a_texcoord, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 
                          (void*)(2 * sizeof(float)));
    
    // Draw full-screen quad (triggers fragment shader for all output pixels)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
    // Ensure GPU is done (synchronize)
    glFinish();
    
    double gpu_end = get_time_ms();
    double gpu_time = gpu_end - gpu_start;
    printf("GPU Time: %.2f ms\n", gpu_time);
    
    // ========================================================================
    // Read back results
    // ========================================================================
    
    unsigned char* result_rgba = (unsigned char*)malloc(dim * dim * 4);
    glReadPixels(0, 0, dim, dim, GL_RGBA, GL_UNSIGNED_BYTE, result_rgba);
    
    // Convert back from RGBA8 to float
    // The shader outputs result/width, so we need to undo that scaling
    // Result is stored in R channel
    for (int i = 0; i < size; i++) {
        // Fragment shader stored (sum / width), so multiply back
        // Then convert from [0,1] (8-bit) back to float
        float normalized = result_rgba[i * 4] / 255.0f;
        C_gpu[i] = normalized * dim;  // Undo the /width in shader
    }
    
    free(result_rgba);
    
    // ========================================================================
    // Verify results
    // ========================================================================
    
    double max_error = 0.0;
    int error_count = 0;
    for (int i = 0; i < size; i++) {
        double error = fabs(C_cpu[i] - C_gpu[i]);
        if (error > max_error) max_error = error;
        // Allow for quantization error (~1% due to 8-bit precision)
        if (error > C_cpu[i] * 0.05 + 0.1) {
            error_count++;
            if (error_count <= 5) {
                printf("  Mismatch at [%d]: CPU=%.4f GPU=%.4f (err=%.4f)\n",
                       i, C_cpu[i], C_gpu[i], error);
            }
        }
    }
    
    printf("\n=========================================\n");
    printf("Results:\n");
    printf("  CPU Time: %.2f ms\n", cpu_time);
    printf("  GPU Time: %.2f ms\n", gpu_time);
    if (gpu_time > 0) {
        printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    }
    printf("  Max Error: %.6f\n", max_error);
    printf("  Results Match: %s\n", error_count == 0 ? "YES" : "NO (see above)");
    if (error_count > 0) {
        printf("  Note: Some error expected due to 8-bit quantization\n");
    }
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

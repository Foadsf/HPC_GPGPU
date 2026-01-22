#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>

// --- Configuration ---
const int WIDTH = 1024;
const int SIZE = WIDTH * WIDTH;

// --- ERROR CALLBACK ---
void glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

std::string loadShader(const char* filename) {
    std::ifstream file(filename);
    if (!file) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void cpu_matrix_mult(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int row = 0; row < WIDTH; row++) {
        for (int col = 0; col < WIDTH; col++) {
            float sum = 0.0f;
            for (int k = 0; k < WIDTH; k++) {
                sum += A[row * WIDTH + k] * B[k * WIDTH + col];
            }
            C[row * WIDTH + col] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    bool skipCPU = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--skip-cpu") == 0) skipCPU = true;
    }

    // 1. Setup Error Callback & Init
    glfwSetErrorCallback(glfw_error_callback);
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // 2. Request Specific Context (4.3 Core) to ensure compatibility
    // NOTE: Raspberry Pi 3B (VideoCore IV) does not natively support 4.3!
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(640, 480, "Hidden", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window. Check your GPU drivers." << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // --- IDENTIFY GPU ---
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* vendor = glGetString(GL_VENDOR);
    std::cout << "=========================================" << std::endl;
    std::cout << "  GPU: " << (renderer ? (const char*)renderer : "Unknown") << std::endl;
    std::cout << "  Vendor: " << (vendor ? (const char*)vendor : "Unknown") << std::endl;
    std::cout << "=========================================" << std::endl;

    // --- DATA GENERATION ---
    std::vector<float> A(SIZE);
    std::vector<float> B(SIZE);
    std::vector<float> C_CPU(SIZE);
    std::vector<float> C_GPU(SIZE);

    for (int i = 0; i < SIZE; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // --- CPU BENCH ---
    if (!skipCPU) {
        std::cout << "Starting CPU Matrix Multiplication..." << std::endl;
        auto startCPU = std::chrono::high_resolution_clock::now();
        cpu_matrix_mult(A, B, C_CPU);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> msCPU = endCPU - startCPU;
        std::cout << "CPU Time: " << msCPU.count() << " ms" << std::endl;
    } else {
        std::cout << "Skipping CPU Benchmark..." << std::endl;
    }

    // --- GPU BENCH ---
    std::string sourceStr = loadShader("compute.glsl");
    if (sourceStr.empty()) {
        std::cerr << "Error: compute.glsl not found!" << std::endl;
        return -1;
    }
    const char* src = sourceStr.c_str();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader Error: " << infoLog << std::endl;
        return -1;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glUseProgram(program);

    std::cout << "Starting GPU Matrix Multiplication..." << std::endl;
    auto startGPU = std::chrono::high_resolution_clock::now();

    GLuint ssboA, ssboB, ssboC;
    glGenBuffers(1, &ssboA); glGenBuffers(1, &ssboB); glGenBuffers(1, &ssboC);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboA);
    glBufferData(GL_SHADER_STORAGE_BUFFER, SIZE * sizeof(float), A.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboA);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboB);
    glBufferData(GL_SHADER_STORAGE_BUFFER, SIZE * sizeof(float), B.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboB);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboC);
    glBufferData(GL_SHADER_STORAGE_BUFFER, SIZE * sizeof(float), NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboC);

    glUniform1i(glGetUniformLocation(program, "WIDTH"), WIDTH);
    glDispatchCompute(WIDTH / 32, WIDTH / 32, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    float* ptr = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (ptr) memcpy(C_GPU.data(), ptr, SIZE * sizeof(float));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> msGPU = endGPU - startGPU;
    std::cout << "GPU Time: " << msGPU.count() << " ms" << std::endl;

    if (!skipCPU) {
        bool correct = true;
        for (int i = 0; i < SIZE; i++) {
            if (std::abs(C_CPU[i] - C_GPU[i]) > 0.1f) { 
                correct = false; break; 
            }
        }
        std::cout << "Results Match: " << (correct ? "YES" : "NO") << std::endl;
    }

    glfwTerminate();
    return 0;
}

#define GLFW_INCLUDE_ES2
#include <GLFW/glfw3.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

// --- Configuration ---
// N=1024 is safe for RGBA8.
const int N = 1024; 

// --- Shader Sources ---
const char* VERT_SRC = R"(
    attribute vec2 position;
    varying vec2 v_texCoord;
    void main() {
        // Map quad (-1..1) to texture coords (0..1)
        v_texCoord = position * 0.5 + 0.5;
        gl_Position = vec4(position, 0.0, 1.0);
    }
)";

// The GPGPU Kernel
// Computes Average: C = Sum(A * B) / N
const char* FRAG_SRC = R"(
    precision mediump float;
    varying vec2 v_texCoord;
    uniform sampler2D texA;
    uniform sampler2D texB;
    uniform float size;

    void main() {
        float sum = 0.0;
        float myCol = v_texCoord.x;
        float myRow = v_texCoord.y;
        float step = 1.0 / size;
        float halfStep = step * 0.5;

        // Loop k from 0 to 1
        // We iterate across the row of A and column of B
        for (float k = 0.0; k < 1.0; k += 0.0009765625) { 
            // 0.0009765625 is 1/1024. Hardcoded for loop stability on legacy compilers.
            
            vec4 valA = texture2D(texA, vec2(k + halfStep, myRow));
            vec4 valB = texture2D(texB, vec2(myCol, k + halfStep));

            // Accumulate Red channel product
            sum += valA.r * valB.r;
        }

        // Divide by N to keep result in 0..1 range (Average)
        // Otherwise RGBA8 will clamp everything > 1.0 to 1.0
        float avg = sum / size;

        gl_FragColor = vec4(avg, 0.0, 0.0, 1.0);
    }
)";

GLuint createShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        std::cerr << "Shader Compile Error: " << log << std::endl;
        exit(1);
    }
    return shader;
}

int main() {
    if (!glfwInit()) return -1;
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(N, N, "GPGPU", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create Context." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    const char* renderer = (const char*)glGetString(GL_RENDERER);
    std::cout << "GL_RENDERER: " << (renderer ? renderer : "Unknown") << std::endl;

    // --- Data Prep ---
    // We use A=1.0 and B=1.0. 
    // Sum = N * 1.0 * 1.0 = 1024.0
    // Average = 1024.0 / 1024.0 = 1.0
    // So expected pixel value is 1.0 (White/Red).
    std::vector<unsigned char> dataA(N * N * 4, 0);
    std::vector<unsigned char> dataB(N * N * 4, 0);

    for (int i = 0; i < N*N; i++) {
        // Red channel = 255 (which is float 1.0)
        dataA[i*4 + 0] = 255; 
        dataA[i*4 + 3] = 255; // Alpha
        
        dataB[i*4 + 0] = 255; 
        dataB[i*4 + 3] = 255; 
    }

    // --- Texture Setup (RGBA8 / Unsigned Byte) ---
    GLuint texA, texB;
    glGenTextures(1, &texA);
    glBindTexture(GL_TEXTURE_2D, texA);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, dataA.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &texB);
    glBindTexture(GL_TEXTURE_2D, texB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, dataB.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // --- FBO Setup ---
    GLuint fbo, texOut;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &texOut);
    glBindTexture(GL_TEXTURE_2D, texOut);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "FBO Incomplete! Driver does not support this format." << std::endl;
        return -1;
    }

    // --- Shader & Draw ---
    GLuint vs = createShader(GL_VERTEX_SHADER, VERT_SRC);
    GLuint fs = createShader(GL_FRAGMENT_SHADER, FRAG_SRC);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glBindAttribLocation(program, 0, "position");
    glLinkProgram(program);
    glUseProgram(program);

    glUniform1i(glGetUniformLocation(program, "texA"), 0);
    glUniform1i(glGetUniformLocation(program, "texB"), 1);
    glUniform1f(glGetUniformLocation(program, "size"), (float)N);

    float vertices[] = { -1.0f, -1.0f,  1.0f, -1.0f,  -1.0f, 1.0f,  1.0f, 1.0f };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glViewport(0, 0, N, N);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texA);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texB);

    std::cout << "Starting GPU Compute..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFinish();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms" << std::endl;

    // --- Readback ---
    std::vector<unsigned char> results(N * N * 4);
    glReadPixels(0, 0, N, N, GL_RGBA, GL_UNSIGNED_BYTE, results.data());

    // Verify Center Pixel
    // We expect 1.0 (which is 255 in bytes)
    int val = results[(N/2 * N + N/2) * 4]; 
    std::cout << "Center Pixel Red Value: " << val << " (Expected: 255)" << std::endl;

    glfwTerminate();
    return 0;
}

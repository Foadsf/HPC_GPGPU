#define GLFW_INCLUDE_ES2
#include <GLFW/glfw3.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>

// --- Configuration ---
const int N = 1024;

// --- Utils ---
std::string loadShaderSource(const char* filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint createShader(GLenum type, const char* filename) {
    std::string source = loadShaderSource(filename);
    const char* src = source.c_str();
    
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        std::cerr << "Shader Error (" << filename << "): " << log << std::endl;
        exit(1);
    }
    return shader;
}

// Check for extension support string
bool checkExtension(const char* extName) {
    const char* extensions = (const char*)glGetString(GL_EXTENSIONS);
    if (!extensions) return false;
    return (strstr(extensions, extName) != NULL);
}

int main() {
    if (!glfwInit()) return -1;
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(N, N, "GPGPU Half-Float", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create Context." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

    // --- Check Extensions ---
    bool hasHalfFloat = checkExtension("GL_OES_texture_half_float");
    std::cout << "GL_OES_texture_half_float: " << (hasHalfFloat ? "FOUND" : "MISSING") << std::endl;
    
    // Some drivers need the specific "Color Buffer" extension to render TO the texture
    bool hasColorBuffer = checkExtension("GL_EXT_color_buffer_half_float"); 
    std::cout << "GL_EXT_color_buffer_half_float: " << (hasColorBuffer ? "FOUND" : "MISSING") << std::endl;

    if (!hasHalfFloat) {
        std::cerr << "Warning: Half-Float not supported. Results will likely clamp to 1.0." << std::endl;
    }

    // --- Data Prep ---
    // A = 1.0, B = 1.0. 
    // Expected Sum = 1024.0.
    std::vector<float> dataA(N * N * 4);
    std::vector<float> dataB(N * N * 4);

    for (int i = 0; i < N*N; i++) {
        dataA[i*4 + 0] = 1.0f; dataA[i*4 + 1] = 0.0f; dataA[i*4 + 2] = 0.0f; dataA[i*4 + 3] = 1.0f;
        dataB[i*4 + 0] = 1.0f; dataB[i*4 + 1] = 0.0f; dataB[i*4 + 2] = 0.0f; dataB[i*4 + 3] = 1.0f;
    }

    // --- Texture Setup ---
    // Define the type. If extension exists, use GL_HALF_FLOAT_OES (0x8D61).
    // Otherwise fallback to GL_FLOAT (which might fail on ES2) or GL_UNSIGNED_BYTE.
    // For this test, we force the attempt.
    GLenum texType = hasHalfFloat ? 0x8D61 : GL_FLOAT; 

    GLuint texA, texB;
    glGenTextures(1, &texA);
    glBindTexture(GL_TEXTURE_2D, texA);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, GL_FLOAT, dataA.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &texB);
    glBindTexture(GL_TEXTURE_2D, texB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, GL_FLOAT, dataB.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // --- FBO Setup (The critical part) ---
    GLuint fbo, texOut;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &texOut);
    glBindTexture(GL_TEXTURE_2D, texOut);
    
    // Attempt to allocate Half-Float texture for output
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, texType, NULL);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texOut, 0);

    GLenum fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fboStatus != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "FBO Error: " << fboStatus << std::endl;
        std::cerr << "Driver likely rejected rendering to Half-Float." << std::endl;
        return -1;
    }

    // --- Shader & Draw ---
    GLuint vs = createShader(GL_VERTEX_SHADER, "vertex.glsl");
    GLuint fs = createShader(GL_FRAGMENT_SHADER, "fragment.glsl");
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

    std::cout << "Starting Half-Float Compute..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFinish();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Time: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms" << std::endl;

    // --- Readback ---
    // We read back as FLOAT. Driver converts Half -> Float.
    std::vector<float> results(N * N * 4);
    glReadPixels(0, 0, N, N, GL_RGBA, GL_FLOAT, results.data());

    float val = results[(N/2 * N + N/2) * 4];
    std::cout << "Center Pixel Value: " << val << std::endl;
    std::cout << "Expected: " << (float)N << ".0" << std::endl;
    
    if (val > 1.0f) {
        std::cout << "SUCCESS: Values > 1.0 preserved. Half-Float GPGPU works!" << std::endl;
    } else {
        std::cout << "FAILURE: Value clamped to 1.0. FBO likely fell back to 8-bit." << std::endl;
    }

    glfwTerminate();
    return 0;
}

#version 430 core

// Launch 32x32 threads per group (1024 threads total per group)
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

 // Matrix Dimensions (passed as a uniform)
uniform int WIDTH;

// Input Buffer A
layout(std430, binding = 0) buffer BufferA {
    float A[];
};

 // Input Buffer B
layout(std430, binding = 1) buffer BufferB {
    float B[];
};

 // Output Buffer C
layout(std430, binding = 2) buffer BufferC {
    float C[];
};

 void main() {
    // Determine which pixel (row, col) this thread is computing
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;

    if (col >= WIDTH || row >= WIDTH) return;

    float sum = 0.0;
    
    // Dot product of Row A and Column B
    for (int k = 0; k < WIDTH; k++) {
        // A is Row-Major: A[row * WIDTH + k]
        // B is Row-Major: B[k * WIDTH + col]
        sum += A[row * WIDTH + k] * B[k * WIDTH + col];
    }

    C[row * WIDTH + col] = sum;
}

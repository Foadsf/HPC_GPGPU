// Fragment shader for OpenGL ES 2.0 GPGPU Matrix Multiplication
// Each fragment computes one element of the output matrix C = A * B
//
// Limitations on VideoCore IV (Pi 3B):
// - No float textures (OES_texture_float not supported)
// - Values are in [0.0, 1.0] range, stored as 8-bit per channel
// - Maximum ~25 texture fetches per fragment (hardware limit)

precision mediump float;

// Input textures containing matrices A and B
// Values are stored as grayscale in R channel, normalized to [0,1]
uniform sampler2D u_matrixA;  // Matrix A (M x K)
uniform sampler2D u_matrixB;  // Matrix B (K x N)

// Matrix dimensions
uniform float u_width;        // Matrix dimension (assuming square matrices)

// Current fragment's texture coordinate
varying vec2 v_texcoord;

void main() {
    // This fragment computes C[row][col]
    // v_texcoord.x maps to column, v_texcoord.y maps to row
    
    float row = floor(v_texcoord.y * u_width);
    float col = floor(v_texcoord.x * u_width);
    
    float sum = 0.0;
    float invWidth = 1.0 / u_width;
    
    // Perform dot product of row from A with column from B
    // Note: This is the inner loop and is the bottleneck
    // On VideoCore IV, we're limited in texture fetches per fragment
    for (float k = 0.0; k < 64.0; k += 1.0) {
        if (k >= u_width) break;
        
        // A[row][k] - fetch from row 'row', column 'k'
        vec2 coordA = vec2((k + 0.5) * invWidth, (row + 0.5) * invWidth);
        float a = texture2D(u_matrixA, coordA).r;
        
        // B[k][col] - fetch from row 'k', column 'col'  
        vec2 coordB = vec2((col + 0.5) * invWidth, (k + 0.5) * invWidth);
        float b = texture2D(u_matrixB, coordB).r;
        
        sum += a * b;
    }
    
    // Clamp result to [0,1] range for storage in RGBA8 texture
    // For values potentially > 1.0, we'd need to scale or use multi-pass
    // This simple version assumes input values are small enough that
    // the product stays in representable range after scaling
    
    // Scale factor: if inputs are in [0,1], max sum = width * 1 * 1 = width
    // So we divide by width to normalize back to [0,1]
    float result = sum / u_width;
    
    // Output: store result in all channels for easier readback
    // Using alpha=1.0 for proper FBO rendering
    gl_FragColor = vec4(result, result, result, 1.0);
}

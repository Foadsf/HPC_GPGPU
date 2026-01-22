// Vertex shader for OpenGL ES 2.0 GPGPU
// Simple pass-through shader that renders a full-screen quad
// The quad triggers fragment shader execution for all output texels

attribute vec2 a_position;  // Vertex position (NDC: -1 to +1)
attribute vec2 a_texcoord;  // Texture coordinate (0 to 1)

varying vec2 v_texcoord;    // Pass to fragment shader

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texcoord = a_texcoord;
}

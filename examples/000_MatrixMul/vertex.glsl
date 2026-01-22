attribute vec2 position;
varying vec2 v_texCoord;

void main() {
    // Map quad (-1..1) to texture coords (0..1)
    v_texCoord = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}

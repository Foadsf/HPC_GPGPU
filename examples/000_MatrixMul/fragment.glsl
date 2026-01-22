precision mediump float;
varying vec2 v_texCoord;

uniform sampler2D texA;
uniform sampler2D texB;
uniform float size;

void main() {
    float sum = 0.0;
    float myCol = v_texCoord.x;
    float myRow = v_texCoord.y;
    
    // We use a fixed step size for N=1024 (1/1024 = 0.0009765625)
    // to avoid precision issues in loop counters on older hardware.
    float step = 0.0009765625;
    float halfStep = step * 0.5;

    for (float k = 0.0; k < 1.0; k += 0.0009765625) { 
        // Read A: Row fixed (myRow), Column moves (k)
        vec4 valA = texture2D(texA, vec2(k + halfStep, myRow));
        
        // Read B: Row moves (k), Column fixed (myCol)
        vec4 valB = texture2D(texB, vec2(myCol, k + halfStep));

        // Multiply & Accumulate (Dot Product)
        sum += valA.r * valB.r;
    }

    // Write the raw sum. 
    // If Half-Float works: Output can be > 1.0.
    // If Half-Float fails: Output will clamp to 1.0.
    gl_FragColor = vec4(sum, 0.0, 0.0, 1.0);
}

////////////////////////////////////////
// diffuse.glsl
////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

#ifdef VERTEX_SHADER

layout(location=0) in vec3 Position;
layout(location=1) in vec2 aTexCoords;

out vec2 In;


////////////////////////////////////////
// Vertex shader
////////////////////////////////////////

void main() {
    In = aTexCoords;
    gl_Position = vec4(Position, 1.0);
}

#endif

////////////////////////////////////////////////////////////////////////////////

#ifdef FRAGMENT_SHADER
precision highp float;
out vec4 FragColor;

in vec2 In;

uniform sampler2D screenTextureR;
uniform sampler2D screenTextureG;
uniform sampler2D screenTextureB;

void main(){
    vec2 uv = In;
    uv = vec2(uv.x,1-uv.y);
    
    float r = clamp(texture(screenTextureR, uv).r,0,1);
    float g = clamp(texture(screenTextureG, uv).r,0,1);
    float b = clamp(texture(screenTextureB, uv).r,0,1);

    vec3 finalColor = vec3(r,g,b);
    finalColor = pow(finalColor, vec3(1.0/2.2));

    FragColor = vec4(finalColor,1.0);


}
#endif

////////////////////////////////////////////////////////////////////////////////

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
layout (location = 0) out vec4 finalColor;
layout (location = 1) out vec4 finalDepth;

in vec2 In;

uniform sampler2D colorTexture;
uniform sampler2D depthTexture;

uniform vec2 viewportSize;
vec2 inverseScreenSize = 1/viewportSize;

const float FXAA_SPAN_MAX = 16.0;
const float FXAA_REDUCE_MUL = 1.0/4.0;
const float FXAA_REDUCE_MIN = 1.0/256.0;


void main(){
    vec2 uv = In;
    finalColor = vec4(texture(colorTexture,uv).rgb,1.0f);
    float depth = texture(depthTexture,uv).r;
    finalDepth = vec4(depth,0,0,1.0f);
//
//
//    vec2 offset = inverseScreenSize;
//    //vec2 uv = In;
//
//    vec3 nw = texture(colorTexture, uv + vec2(-1.0, -1.0) * offset).rgb;
//    vec3 ne = texture(colorTexture, uv + vec2( 1.0, -1.0) * offset).rgb;
//    vec3 sw = texture(colorTexture, uv + vec2(-1.0,  1.0) * offset).rgb;
//    vec3 se = texture(colorTexture, uv + vec2( 1.0,  1.0) * offset).rgb;
//    vec3 m  = texture(colorTexture, uv).rgb;
//
//    vec3 luma = vec3(0.299, 0.587, 0.114);
//    float lumaNW = dot(nw, luma);
//    float lumaNE = dot(ne, luma);
//    float lumaSW = dot(sw, luma);
//    float lumaSE = dot(se, luma);
//    float lumaM  = dot(m,  luma);
//
//    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
//    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
//    vec2 dir = vec2(
//        -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
//        ((lumaNW + lumaSW) - (lumaNE + lumaSE)));
//
//    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
//    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
//    dir = min(vec2(FXAA_SPAN_MAX), max(vec2(-FXAA_SPAN_MAX), dir * rcpDirMin)) * offset;
//
//    vec3 rgbA = 0.5 * (texture(colorTexture, uv + dir * (1.0 / 3.0 - 0.5)).xyz + texture(colorTexture, uv + dir * (2.0 / 3.0 - 0.5)).xyz);
//    vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(colorTexture, uv + dir * -0.5).xyz + texture(colorTexture, uv + dir * 0.5).xyz);
//    float lumaB = dot(rgbB, luma);
//    if (lumaB < lumaMin || lumaB > lumaMax) {
//        finalColor = vec4(rgbA, 1.0);
//    } else {
//        finalColor = vec4(rgbB, 1.0);
//    }
}
#endif

////////////////////////////////////////////////////////////////////////////////

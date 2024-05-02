////////////////////////////////////////
// diffuse.glsl
////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

#ifdef VERTEX_SHADER

layout(location=0) in vec3 Position;
layout(location=1) in vec3 Normal;

out vec3 fragPosition;
out vec3 fragNormal;

uniform mat4 ModelMtx=mat4(1);
uniform mat4 ModelViewProjMtx=mat4(1);
uniform mat4 InvTransModelMtx=mat4(1);

////////////////////////////////////////
// Vertex shader
////////////////////////////////////////

void main() {
    
	gl_Position= ModelViewProjMtx * vec4(Position,1);
    
    vec4 out_pos = ModelMtx * vec4(Position,1);

	fragPosition= out_pos.xyz/out_pos.w;
	fragNormal= normalize(vec3(ModelMtx * vec4(Normal,0)));
}

#endif

////////////////////////////////////////////////////////////////////////////////

#ifdef FRAGMENT_SHADER

in vec3 fragPosition;
in vec3 fragNormal;

// material parameters
uniform vec3 albedo = vec3(1,1,1);
uniform vec3 reflectance = vec3(1.0,1.0,1.0);
uniform float roughness = 0.04;

// lights
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];
vec3 lightPoints[4];
uniform vec3 camPos;
uniform bool twoSided = true;
uniform vec3 lightPositionDelta = vec3(0,0,0);;
uniform float uTime;
layout (location = 0) out vec3 gNormal;
layout (location = 1) out vec3 gPosition;
layout (location = 2) out vec4 gMaterial_1;
layout (location = 3) out vec4 gMaterial_2;
layout (location = 4) out vec3 gView;

const float PI = 3.14159265359;

void main() 
{
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(camPos - fragPosition);
    gNormal = N;
    gPosition = fragPosition;
    gMaterial_1 = vec4(reflectance,roughness);
    gMaterial_2 = vec4(0);
    gView = V;
}

#endif

////////////////////////////////////////////////////////////////////////////////

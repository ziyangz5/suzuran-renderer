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
	gl_Position=ModelViewProjMtx * vec4(Position,1);

	fragPosition=vec3(ModelMtx * vec4(Position,1));
	fragNormal=vec3(InvTransModelMtx * vec4(Normal,0));
}

#endif

////////////////////////////////////////////////////////////////////////////////

#ifdef FRAGMENT_SHADER

in vec3 fragPosition;
in vec3 fragNormal;

// material parameters
uniform vec3 albedo = vec3(1,1,1);
uniform vec3 reflectance = vec3(1.0,1.0,1.0);

// lights
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

uniform vec3 camPos;
uniform float uTime;

out vec4 finalColor;

const float PI = 3.14159265359;
// 

void main() 
{
    vec3 N = normalize(fragNormal);
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) 
    {
        vec3 L = normalize(lightPositions[i] - fragPosition);
        float distance = length(lightPositions[i] - fragPosition);
        float attenuation = 1000.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;
        float NdotL = max(dot(N, L), 0.0); 
        Lo += reflectance * radiance * NdotL;
    }   

    vec3 ambient = vec3(0.03) * albedo;

    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2)); 

    finalColor = vec4(color, 1.0);
    //finalColor = vec4(1,0,1, 1.0);
}

#endif

////////////////////////////////////////////////////////////////////////////////

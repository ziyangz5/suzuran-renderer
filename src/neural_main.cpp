#include <torch/script.h>
#include <glad/glad.h>
#include "Core.h"
#include "Camera.h"
#include "Shader.h"
#include "Scene.h"
#include <sstream>
#include <fstream>
#include "SceneParser.h"
#include "ltc.h"
#include <glm/gtx/string_cast.hpp>

//#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include "cudahelperlib.h"
#include "Postprocess.h"
#include "VariableSceneConfig.h"

#include <random>

using namespace szr;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void processInput(GLFWwindow *window, Scene*scene);
void showFPS(GLFWwindow* pWindow);
Camera camera(glm::vec3(0, 1.0f, 7.0f));
float lastX = 800.0f / 2.0;
float lastY = 600.0 / 2.0;
bool firstMouse = true;
float deltaTime = 0.0f;
long nbFrames = 0;
float lastTime = 0.0f;
int WinX = 1024; int WinY = 768;
bool mouseMoving = false;
bool requestResizeGbuffer = false;
int requestSaveFramebuffer = 0;
const std::string savePath = "./savedImages/";
unsigned int quadVAO = 0;
unsigned int quadVBO;
VariableSceneConfig* sconfig;
//float param0 = 1.0;
//float param1 = 0.5;
//float param2 = 0.5;
//float param3 = 0.5;
//float param4 = 0.5;
//float param5 = 0.0;

void translate_shape(Scene* scene, const char* name, float x, float y,float z)
{
    for (uint i = 0; i < scene->glass_models.size(); i++)
    {
        if (scene->glass_models[i]->tag != name) continue;
        scene->glass_models[i]->SetWorldMatrix(glm::identity<glm::mat4>());
        scene->glass_models[i]->ApplyTransformMatrix(glm::translate(glm::vec3(x,y,z)));
        break;
    }
}

void set_variable(Scene* scene,const char* name, const char* key, float value)
{
    for (uint i = 0; i < scene->models.size(); i++)
    {
        if (scene->models[i]->tag != name) continue;
        ShaderProgram* geo_program = scene->geo_shaders[scene->models[i]->shaderID];
        if (geo_program != nullptr)
        {
            geo_program->use();
            geo_program->setFloat(key,value);
            geo_program->unuse();
        }
    }
}



void set_camera_transform(Scene* scene, glm::vec3 pos,glm::vec3 target,glm::vec3 up)
{
    glm::mat<4,4,float> to_world = glm::lookAt(pos, target, up);

    glm::mat4 inverted = glm::inverse(to_world);
    const glm::vec3 direction = -glm::vec3(inverted[2]);
    float yaw = glm::degrees(glm::atan(direction.z, direction.x));
    float pitch = glm::degrees(glm::asin(direction.y));

    scene->camera.SetTransform(glm::vec2(pitch,yaw),pos);

}

void ConfigGBuffer(uint& gNormal, uint& gPosition, uint& gMaterial_1, uint& gMaterial_2, uint& gView, uint& rboDepth)
{
    glGenTextures(1, &gNormal);
    glBindTexture(GL_TEXTURE_2D, gNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WinX, WinY, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gNormal, 0);

    glGenTextures(1, &gPosition);
    glBindTexture(GL_TEXTURE_2D, gPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WinX, WinY, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gPosition, 0);

    glGenTextures(1, &gMaterial_1);
    glBindTexture(GL_TEXTURE_2D, gMaterial_1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WinX, WinY, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gMaterial_1, 0);

    glGenTextures(1, &gMaterial_2);
    glBindTexture(GL_TEXTURE_2D, gMaterial_2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WinX, WinY, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, gMaterial_2, 0);

    glGenTextures(1, &gView);
    glBindTexture(GL_TEXTURE_2D, gView);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WinX, WinY, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, gView, 0);

    glGenTextures(1, &rboDepth);
    glBindTexture(GL_TEXTURE_2D, rboDepth);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, WinX, WinY, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, rboDepth, 0);

//    glGenRenderbuffers(1, &rboDepth);
//    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
//    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WinX, WinY);
//    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
    // finally check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;

}

void PostProcessingRenderingDepthPass(ShaderProgram* ppShader, uint& pRenderingResult, uint& pDepthBuffer)
{
    if (quadVAO == 0)
    {
        float quadVertices[] = {
                // positions        // texture Coords
                -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
                -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
                1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
                1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    ppShader->use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, pRenderingResult);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, pDepthBuffer);
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    ppShader->unuse();
}

void PostProcessingRendering(ShaderProgram* ppShader, uint& pRenderingResultR, uint& pRenderingResultG, uint& pRenderingResultB)
{
    glDrawBuffer(GL_BACK);
    if (quadVAO == 0)
    {
        float quadVertices[] = {
                // positions        // texture Coords
                -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
                -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
                1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
                1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    ppShader->use();
    ppShader->setVec2("viewportSize", WinX, WinY);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, pRenderingResultR);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, pRenderingResultG);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, pRenderingResultB);
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    ppShader->unuse();
}

void CreateCUDAResource( cudaGraphicsResource_t& cudaResource, GLuint GLtexture, cudaGraphicsMapFlags mapFlags )
{
    // Map the GL texture resource with the CUDA resource
    cudaErrorCheck(cudaGraphicsGLRegisterImage( &cudaResource, GLtexture, GL_TEXTURE_2D, mapFlags ));
}

void ConfigPBuffer(uint& pRenderingResult, uint& pDepthBuffer)
{
    glGenTextures(1, &pRenderingResult);
    glBindTexture(GL_TEXTURE_2D, pRenderingResult);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WinX, WinY, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pRenderingResult, 0);

    glGenRenderbuffers(1, &pDepthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, pDepthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WinX, WinY);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, pDepthBuffer);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;
}

void ConfigPBufferDepthPass(uint& pRenderingResultDepthPass, uint& pDepthDataDepthPass, uint& pDepthBufferDepthPass)
{
    glGenTextures(1, &pRenderingResultDepthPass);
    glBindTexture(GL_TEXTURE_2D, pRenderingResultDepthPass);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WinX, WinY, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pRenderingResultDepthPass, 0);

    glGenTextures(1, &pDepthDataDepthPass);
    glBindTexture(GL_TEXTURE_2D, pDepthDataDepthPass);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WinX, WinY, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, pDepthDataDepthPass, 0);

    glGenRenderbuffers(1, &pDepthBufferDepthPass);
    glBindRenderbuffer(GL_RENDERBUFFER, pDepthBufferDepthPass);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WinX, WinY);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, pDepthBufferDepthPass);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;
}



// Create a texture resource for rendering to.
void CreateTexture( GLuint& texture, unsigned int width, unsigned int height )
{
    glGenTextures( 1, &texture );
    glBindTexture( GL_TEXTURE_2D, texture );

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Create texture data (4-component unsigned byte)
    glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, NULL );

    // Unbind the texture
    glBindTexture( GL_TEXTURE_2D, 0 );
}


void render(GLFWwindow* window, Scene* scene)
{
    torch::jit::script::Module cnet;
    torch::jit::script::Module nsrr;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        //cnet = torch::jit::load("../neural_model/traced_crystalnet_at.pt");
        //cnet = torch::jit::load("../neural_model/traced_super_crystalnet_bathroom.pth");
        //cnet = torch::jit::load("../neural_model/traced_super_crystalnet_cornellbox8.pth");
        //cnet = torch::jit::load("../neural_model/traced_super_crystalnet_cornellbox_2x.pth");
        cnet = torch::jit::load("../neural_model/traced_super_crystalnet_cornellbox_new.pth");
        nsrr = torch::jit::load("../neural_model/NSRR_cornellbox_2x.pth");
        //cnet = torch::jit::load("../neural_model/traced_super_crystalnet_desk.pth");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cerr <<  e.msg();
        exit(1);
    }

    std::cout << "CrystalNet Loaded.\n";
    cnet.to(torch::kCUDA, torch::kFloat32);
    nsrr.to(torch::kCUDA, torch::kFloat32);
    std::cout << "CrystalNet moved to GPU.\n";
    cnet.eval();
    nsrr.eval();

    sconfig = new VariableSceneConfig("super_cbox",scene);

    //bathroom
//    glm::vec3 min_bounds(-10, 17, 20);
//    glm::vec3 range_bounds(20, 2, 25);
//    glm::vec3 bbox_min(-20.143, -2.4551, -22.9468);
//    glm::vec3 bbox_range(39.1102, 39.4102, 76.4793);
    //cbox
    glm::vec3 min_bounds(-10, 20, -40);
    glm::vec3 range_bounds(10, 10, 20);
    glm::vec3 bbox_min(-25.082143783569336, -5.054699897766113, -11.865564346313477);
    glm::vec3 bbox_range(57.182472229003906, 56.89321517944336, 57.511619567871094);
    //cbox8
//    glm::vec3 min_bounds(-10, 20, -40);
//    glm::vec3 range_bounds(9, 9, 20);
//    glm::vec3 bbox_min(-25.082143783569336, -5.054699897766113, -11.865564346313477);
//    glm::vec3 bbox_range(57.182472229003906, 56.89321517944336, 57.511619567871094);
    //desk
//    glm::vec3 min_bounds(-9, 8.0, -7.5);
//    glm::vec3 range_bounds(10.5, 5.5, 15.5);
//    glm::vec3 bbox_min(-10.779967308044434, -5.381340980529785, -9.664804458618164);
//    glm::vec3 bbox_range(20.77996826171875, 20.000003814697266, 20.776588439941406);


    float lastFrame = 0.0f;

    uint ltcMatId;
    float ltc_mat[64 * 64 * 4];
    uint ltcMagId;
    float ltc_mag[64 * 64];

    uint gBuffer, pBuffer, pBufferDepthPass;
    //Initializing g-buffer
    glGenFramebuffers(1, &gBuffer);
    uint gNormal, gPosition, gMaterial_1, gMaterial_2, gView, gLightDir, rboDepth, screenBuffer;
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    ConfigGBuffer(gNormal, gPosition, gMaterial_1, gMaterial_2, gView, rboDepth);
    uint gAttachments[5] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_COLOR_ATTACHMENT3,GL_COLOR_ATTACHMENT4};
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    uint tgBuffer;
    glGenFramebuffers(1, &tgBuffer);
    uint tgNormal, tgPosition, tgMaterial_1, tgMaterial_2, tgView, trboDepth;
    glBindFramebuffer(GL_FRAMEBUFFER, tgBuffer);
    ConfigGBuffer(tgNormal, tgPosition, tgMaterial_1, tgMaterial_2, tgView, trboDepth);
    uint tgAttachments[5] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_COLOR_ATTACHMENT3,GL_COLOR_ATTACHMENT4};
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //Initializing depth acquiring pass buffer
    ShaderProgram* ppDepthPass = new ShaderProgram("../shaders/depth_acquire.glsl", ShaderProgram::eRender);
    ppDepthPass->use();
    ppDepthPass->setInt("colorTexture", 0);
    ppDepthPass->setInt("depthTexture", 1);
    ppDepthPass->setVec2("viewportSize", WinX, WinY);
    ppDepthPass->unuse();
    glGenFramebuffers(1, &pBufferDepthPass);
    uint pRenderingResultDepthPass,pDepthDataDepthPass, pDepthBufferDepthPass;
    glBindFramebuffer(GL_FRAMEBUFFER, pBufferDepthPass);
    ConfigPBufferDepthPass(pRenderingResultDepthPass, pDepthDataDepthPass,pDepthBufferDepthPass);
    uint pAttachmentsDepthPass[2] = { GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1 };
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    //Initializing post-processing-buffer
//    ShaderProgram* ppFXAA = new ShaderProgram("../shaders/pp_FXAA.glsl", ShaderProgram::eRender);
//    ppFXAA->use();
//    ppFXAA->setInt("screenTexture", 0);
//    ppFXAA->unuse();
//    glGenFramebuffers(1, &pBuffer);
//    uint pRenderingResult, pDepthBuffer;
//    glBindFramebuffer(GL_FRAMEBUFFER, pBuffer);
//    ConfigPBuffer(pRenderingResult, pDepthBuffer);
//    uint pAttachments[1] = { GL_COLOR_ATTACHMENT0 };
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    ShaderProgram* ppFXAA = new ShaderProgram("../shaders/pp_CUDA.glsl", ShaderProgram::eRender);
    ppFXAA->use();
    ppFXAA->setInt("screenTextureR", 0);
    ppFXAA->setInt("screenTextureG", 1);
    ppFXAA->setInt("screenTextureB", 2);
    ppFXAA->unuse();
    glGenFramebuffers(1, &pBuffer);
    uint pRenderingResult, pDepthBuffer;
    glBindFramebuffer(GL_FRAMEBUFFER, pBuffer);
    ConfigPBuffer(pRenderingResult, pDepthBuffer);
    uint pAttachments[1] = { GL_COLOR_ATTACHMENT0 };
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    cudaGraphicsResource_t cNormal, cPosition, cMaterial_1, cMaterial_2, cView, cResult, cDepth, nnResultR, nnResultG, nnResultB;
    cudaGraphicsResource_t tcNormal, tcPosition, tcMaterial_1, tcMaterial_2, tcView, tcDepth;

    uint cudaResultR,cudaResultG,cudaResultB;
    CreateTexture( cudaResultR, WinX, WinY );
    CreateTexture( cudaResultG, WinX, WinY );
    CreateTexture( cudaResultB, WinX, WinY );

    CreateCUDAResource( cNormal, gNormal, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( cPosition, gPosition, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( cMaterial_1, gMaterial_1, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( cMaterial_2, gMaterial_2, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( cView, gView, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( cResult, pRenderingResult, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( cDepth, pDepthDataDepthPass, cudaGraphicsMapFlagsReadOnly );

    CreateCUDAResource( tcNormal, tgNormal, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( tcPosition, tgPosition, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( tcMaterial_1, tgMaterial_1, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( tcMaterial_2, tgMaterial_2, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( tcView, tgView, cudaGraphicsMapFlagsReadOnly );
    CreateCUDAResource( tcDepth, pDepthDataDepthPass, cudaGraphicsMapFlagsReadOnly );


    CreateCUDAResource( nnResultR, cudaResultR, cudaGraphicsMapFlagsWriteDiscard );
    CreateCUDAResource( nnResultG, cudaResultG, cudaGraphicsMapFlagsWriteDiscard );
    CreateCUDAResource( nnResultB, cudaResultB, cudaGraphicsMapFlagsWriteDiscard );

//    float g_CurrentFilter[] = {
//            0, 0, 0,  0, 0,
//            0, 0, 0,  0, 0,
//            0, 0, 1,  0, 0,
//            0, 0, 0, -1, 0,
//            0, 0, 0,  0, 0,
//            1, 128
//    };
    float g_CurrentFilter[] = {
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 0
    };
    float g_Scale = g_CurrentFilter[25];
    float g_Offset = g_CurrentFilter[26];
    std::mt19937 gen(42);
    std::vector<std::tuple<glm::vec3, glm::vec2, glm::vec3,std::vector<glm::vec3>>> record;

    //Initializing shader config - lighting
    for (unsigned int sidx = 0; sidx < scene->shaders.size(); sidx++)
    {
        ShaderProgram* program = scene->shaders[sidx];
        program->use();
        //for (uint i = 0; i < scene->lights.size(); ++i)
        //{
        //    program->setVec3("lightPositions[" + std::to_string(i) + "]", scene->lights[i]->position);
        //}
        //TEMP: Dirty way to set light
        //TODO: Support multiple area emitters
        for (int i = 0; i < scene->lights.size(); i++)
        {
            program->setVec3("lightPositions[" + std::to_string(0 + i * 4) + "]", scene->lights[i]->pos_x1 + scene->light_position_shift[i]);
            program->setVec3("lightPositions[" + std::to_string(1 + i * 4) + "]", scene->lights[i]->pos_x2 + scene->light_position_shift[i]);
            program->setVec3("lightPositions[" + std::to_string(2 + i * 4) + "]", scene->lights[i]->pos_x3 + scene->light_position_shift[i]);
            program->setVec3("lightPositions[" + std::to_string(3 + i * 4) + "]", scene->lights[i]->pos_x4 + scene->light_position_shift[i]);
            program->setVec3("lightColors[" + std::to_string(i ) + "]", scene->lights[i]->color);
            program->setBool("twoSided[" + std::to_string(i ) + "]", scene->lights[i]->two_sided);
        }
        program->setInt("numberOfLights", scene->lights.size());

        if (program->name == "area_ggx" || program->name == "glass")
        {
            std::copy(std::begin(g_ltc_mat), std::end(g_ltc_mat), std::begin(ltc_mat));
            glGenTextures(1, &ltcMatId);
            glBindTexture(GL_TEXTURE_2D, ltcMatId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 64, 64, 0, GL_RGBA, GL_FLOAT,
                         (void*)(&ltc_mat));

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            std::copy(std::begin(g_ltc_mag), std::end(g_ltc_mag), std::begin(ltc_mag));
            glGenTextures(1, &ltcMagId);
            glBindTexture(GL_TEXTURE_2D, ltcMagId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 64, 64, 0, GL_RED, GL_FLOAT,
                         (void*)(&ltc_mag));

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);

            glActiveTexture(GL_TEXTURE0);
            program->setInt("ltc_mat", 0);
            glBindTexture(GL_TEXTURE_2D, ltcMatId);

            glActiveTexture(GL_TEXTURE0 + 1);
            program->setInt("ltc_mag", 1);
            glBindTexture(GL_TEXTURE_2D, ltcMagId);
        }
        program->unuse();
        glActiveTexture(GL_TEXTURE0);
    }

    int num_of_glass = scene->glass_models.size();
    std::cout<<"Num of glass: "<<num_of_glass<<std::endl;

    glm::vec3 camera_translate; glm::vec2 camera_euler_delta; glm::vec3 lightPositionDelta; std::vector<glm::vec3> glassesPositionDelta;

    //Game loop
    long long counter = 0;
    while (!glfwWindowShouldClose(window)) {
        counter++;
        float currentFrame = static_cast<float>(glfwGetTime());

        processInput(window,scene);
        deltaTime = currentFrame - lastFrame;
        if (counter > 128)
        {
            sconfig->process_transformation(scene,deltaTime);
        }

        lastFrame = currentFrame;

        for (unsigned int sidx = 0; sidx < scene->shaders.size(); sidx++)
        {
            ShaderProgram* program = scene->shaders[sidx];
            program->use();
            //TEMP: Dirty way to set light
            for (int i = 0; i < scene->lights.size(); i++)
            {
                program->setVec3("lightPositions[" + std::to_string(0 + i * 4) + "]", scene->lights[i]->pos_x1 + scene->light_position_shift[i]);
                program->setVec3("lightPositions[" + std::to_string(1 + i * 4) + "]", scene->lights[i]->pos_x2 + scene->light_position_shift[i]);
                program->setVec3("lightPositions[" + std::to_string(2 + i * 4) + "]", scene->lights[i]->pos_x3 + scene->light_position_shift[i]);
                program->setVec3("lightPositions[" + std::to_string(3 + i * 4) + "]", scene->lights[i]->pos_x4 + scene->light_position_shift[i]);
                program->setVec3("lightColors[" + std::to_string(i ) + "]", scene->lights[i]->color);
                program->setBool("twoSided[" + std::to_string(i ) + "]", scene->lights[i]->two_sided);
            }
        }

        glm::vec3 origin = min_bounds + glm::vec3(sconfig->params[0],sconfig->params[1],sconfig->params[2]) * range_bounds;
        glm::vec3 target(bbox_min[0] + bbox_range[0] / 3 + (sconfig->params[3] * bbox_range[0] / 3),
                         bbox_min[1] + (bbox_range[1] / 3),
                         bbox_min[2] + bbox_range[2] / 3 + (sconfig->params[4] * bbox_range[2] / 3));

        //std::cout<<glm::to_string(origin)<<std::endl;
        //std::cout<<glm::to_string(target)<<std::endl;

        set_camera_transform(scene,origin,target,glm::vec3(0,1,0));
        camera = scene->camera;
        showFPS(window);

        //Geo Pass
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
        glDrawBuffers(5, gAttachments);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1);
        for (uint i = 0; i < scene->models.size(); i++)
        {
            ShaderProgram* geo_program = scene->geo_shaders[scene->models[i]->shaderID];
            if (geo_program != nullptr)
            {
                if (geo_program->name == "glass_geo")
                {
                    continue;
                }
                geo_program->use();
                if (geo_program->has_texture)
                {
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, geo_program->getTexture());
                    geo_program->setInt("ambient_texture", 0);
                }
                geo_program->setVec3("lightPositionDelta", lightPositionDelta);
                Model* model = scene->models[i];
                model->Draw(camera.GetViewProjectionMatrix(WinX, WinY), camera.Position, geo_program, currentFrame);
                geo_program->unuse();
            }
        }
        //cNormal, cPosition, cMaterial_1, cMaterial_2, cView, cResult;
        torch::Tensor tNormal = torch::flipud(CreateTensorFromGLViaCUDA(cNormal,WinX,WinY)).clone();
        torch::Tensor tPosition = torch::flipud(CreateTensorFromGLViaCUDA(cPosition,WinX,WinY));
        torch::Tensor tMaterial_1 = torch::flipud(CreateTensorFromGLViaCUDA(cMaterial_1,WinX,WinY));
        // std::cout << torch::sum(torch::nonzero(torch::isnan(tMaterial_1))) << std::endl;
        //torch::Tensor tMaterial_2 =  torch::zeros({220, 220, 1}).to(torch::kFloat32).to(torch::kCUDA);
        torch::Tensor tView = torch::flipud(CreateTensorFromGLViaCUDA(cView,WinX,WinY));
        //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //glClearColor(0, 0, 0, 1);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);



        //torch::Tensor ttDepthSliced = torch::zeros({WinX, WinY, 1}).to(torch::kFloat32).to(torch::kCUDA);
        torch::Tensor Xg = torch::zeros({1, 0, WinX, WinY, 17},torch::kCUDA).to(torch::kFloat32);
        for (uint i = 0; i < scene->glass_models.size(); i++)
        {
            glDisable(GL_BLEND);
            glBindFramebuffer(GL_FRAMEBUFFER, tgBuffer);
            glDrawBuffers(5, tgAttachments);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0, 0, 0, 1);
            ShaderProgram* geo_program = scene->geo_shaders[scene->glass_models[i]->shaderID];
            if (geo_program != nullptr)
            {
                geo_program->use();
                Model* model = scene->glass_models[i];
                model->Draw(camera.GetViewProjectionMatrix(WinX, WinY), camera.Position, geo_program, currentFrame);
                geo_program->unuse();
            }
            torch::Tensor ttNormal = torch::flipud(CreateTensorFromGLViaCUDA(tcNormal,WinX,WinY));
            torch::Tensor ttPosition = torch::flipud(CreateTensorFromGLViaCUDA(tcPosition,WinX,WinY));
            torch::Tensor ttMaterial_1 = torch::flipud(CreateTensorFromGLViaCUDA(tcMaterial_1,WinX,WinY)) ;
            torch::Tensor ttView = torch::flipud(CreateTensorFromGLViaCUDA(tcView,WinX,WinY));
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, pBufferDepthPass);
            glDrawBuffers(2, pAttachmentsDepthPass);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0, 0, 0, 1);

            PostProcessingRenderingDepthPass(ppDepthPass,pRenderingResult,rboDepth);
            torch::Tensor ttDepthSliced = CreateTensorFromGLViaCUDA(tcDepth,WinX,WinY).index({"...", torch::indexing::Slice(torch::indexing::None, 1) });

            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glBlitFramebuffer(0, 0, WinX, WinY, 0, 0, WinX, WinY, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            torch::Tensor singleXg = torch::concat({ttNormal,ttPosition,ttMaterial_1,ttView,ttDepthSliced},2).view({1, 1, WinX, WinY,17});
            Xg = torch::concat({singleXg,Xg},1);
        }
        //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //cNormal, cPosition, cMaterial_1, cMaterial_2, cView, cResult;


        //glClearColor(0, 0, 0, 1);




        //Foward Pass
        //glEnable(GL_BLEND);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindFramebuffer(GL_FRAMEBUFFER, pBuffer);
        glDrawBuffers(1, pAttachments);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1);

        for (uint i = 0; i < scene->models.size(); i++)
        {
            ShaderProgram* program = scene->shaders[scene->models[i]->shaderID];

            program->use();
            if (program->name == "area_ggx")
            {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, ltcMatId);

                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, ltcMagId);

                if (program->has_texture)
                {
                    glActiveTexture(GL_TEXTURE2);
                    glBindTexture(GL_TEXTURE_2D, program->getTexture());
                    program->setInt("ambient_texture", 2);
                }

            }
            Model* model = scene->models[i];
            model->Draw(camera.GetViewProjectionMatrix(WinX, WinY), camera.Position, program, currentFrame);

            program->unuse();
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //Depth-Acquire Pass
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, pBufferDepthPass);
        glDrawBuffers(2, pAttachmentsDepthPass);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1);

        PostProcessingRenderingDepthPass(ppDepthPass,pRenderingResult,rboDepth);
        torch::Tensor tDepthSliced = CreateTensorFromGLViaCUDA(cDepth,WinX,WinY).index({"...", torch::indexing::Slice(torch::indexing::None, 1) });

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, WinX, WinY, 0, 0, WinX, WinY, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);



        //Post-Processing Pass
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawBuffer(GL_BACK);
//        PostProcessingRendering(ppFXAA,pRenderingResult);
//        PostprocessCUDA( gCUDAResourceDst, gCUDAResourceSrc, WinX, WinY, g_CurrentFilter, g_Scale, g_Offset, counter==30);
//cNormal, cPosition, cMaterial_1, cMaterial_2, cView, cResult;
        torch::Tensor tResult = torch::flipud(CreateTensorFromGLViaCUDA(cResult,WinX,WinY));



//        std::cout<<torch::sum(tResult)<<std::endl;
        torch::Tensor X = torch::concat({tNormal,tPosition,tMaterial_1,tView,tDepthSliced,tResult},2).view({1, WinX, WinY,21});

        X = torch::moveaxis(X,3,1);

//        std::cout<<torch::mean(tNormal).item<float>() << " "
//                <<torch::mean(tPosition).item<float>() << " "
//                <<torch::mean(tMaterial_1).item<float>() << " "
//                <<torch::mean(tView).item<float>() << " "
//                <<torch::mean(tDepthSliced).item<float>() << " "
//                <<torch::mean(tResult).item<float>() << " " <<std::endl;



        //torch::Tensor Xg = torch::concat({ttNormal,ttPosition,ttMaterial_1,ttView,ttDepthSliced},2).view({1,1, WinX, WinY,17});//torch::zeros({1, 1, 17, 256, 256}).to(torch::kFloat32).to(torch::kCUDA);
        //torch::Tensor Xg = torch::zeros({1, 1, 17, 256, 256}).to(torch::kFloat32).to(torch::kCUDA);


        Xg = torch::moveaxis(Xg,4,2);
        //torch::Tensor ld = torch::;
        std::vector<torch::jit::IValue> inputs{X,Xg};
//        inputs.emplace_back((X));
//        inputs.emplace_back((Xg));
//        if (WinX == 256)
//        {
//            inputs.push_back(nan_to_num(X.index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
//                                                 torch::indexing::Slice(-4, -1),
//                                                 torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
//                                                 torch::indexing::Slice(torch::indexing::None, torch::indexing::None)})));
//        }





        torch::Tensor img_view = torch::zeros( {1, 3, 6, WinX, WinY},torch::kCUDA).to(torch::kFloat32);
        torch::Tensor img_depth = torch::zeros({1, 1, 6, WinX, WinY},torch::kCUDA).to(torch::kFloat32);
        torch::Tensor img_flow = torch::zeros( {1, 2, 6, WinX, WinY},torch::kCUDA).to(torch::kFloat32);

//
        torch::Tensor final_result = torch::pow(cnet.forward(inputs).toTensor(),2.2f);

        std::vector<torch::jit::IValue> inputs_nsrr{img_view,img_depth,img_flow};
        torch::Tensor up_result = nsrr.forward(inputs_nsrr).toTensor();
        // std::cout<<up_result.sizes()<<std::endl;
        //torch::Tensor final_result = cnet.forward(inputs).toTensor();
        final_result = torch::moveaxis(final_result,1,3).view({WinX,WinY,3});

//        if (counter == 100)
//        {
//            std::ofstream myfile ("example.txt");
//            std::cout<<"Writing"<<std::endl;
//            if (myfile.is_open())
//            {
//                for(int x = 0; x < WinX; x ++){
//                    for(int y = 0; y < WinY; y ++){
//                        for(int z = 0; z < 4; z ++){
//                            if (z == 3)
//                            {
//                                myfile<< 1<< ",";
//                            }
//                            else
//                            {
//                                myfile<< tResult[x][y][z].item<float>() << ",";
//                            }
//                        }
//                    }
//                }
//                myfile.close();
//            }
//            std::cout<<"Wrote"<<std::endl;
//        }
        torch::Tensor final_result_r = final_result.index({"...",0});
        torch::Tensor final_result_g = final_result.index({"...",1});
        torch::Tensor final_result_b = final_result.index({"...",2});

        float* rb_r = MoveDataFromTensorToGLTexture(nnResultR,final_result_r,WinX,WinY);
        float* rb_g = MoveDataFromTensorToGLTexture(nnResultG,final_result_g,WinX,WinY);
        float* rb_b = MoveDataFromTensorToGLTexture(nnResultB,final_result_b,WinX,WinY);
//        cudaFree(rb_r);
        cudaFree(rb_g);
        cudaFree(rb_b);




        PostProcessingRendering(ppFXAA,cudaResultR,cudaResultG,cudaResultB);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, WinX, WinY, 0, 0, WinX, WinY, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();

    }

}

int main(int argc, char *argv[]) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WinX, WinY, "test", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval( 0 );
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, WinX, WinY);
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);
    //glEnable(GL_FRAMEBUFFER_SRGB);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    printf("OpenGL version used by this application (%s): \n", glGetString(GL_VERSION));

    //Scene* scene = SceneParser::parse_scene("../Scenes/cbox_t/cbox_opengl_area.xml");
    //Scene* scene = SceneParser::parse_scene("../Scenes/bathroom/bathroom_opengl_area.xml");
    Scene* scene = SceneParser::parse_scene("../Scenes/cornellbox/cornellbox_opengl_area.xml");
    //Scene* scene = SceneParser::parse_scene("../Scenes/cornellbox8/cornellbox8_opengl_area.xml");
    //Scene* scene = SceneParser::parse_scene("../Scenes/desk/desk_opengl_area.xml");
    camera = scene->camera;
    WinX = camera.defaultWinX;
    WinY = camera.defaultWinY;
    glfwSetWindowSize(window, WinX, WinY);
    glViewport(0, 0, WinX, WinY);

    int device_count;
    cudaErrorCheck(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "CUDA Error: No cuda device found");
    }
    else
    {
        cudaErrorCheck(cudaSetDevice(0));
        std::cout <<"CUDA initialized with device count = " << device_count << std::endl;
    }



    render(window, scene);

    glfwTerminate();
    return 0;
}

void showFPS(GLFWwindow* pWindow)
{
    // Measure speed
    double currentTime = glfwGetTime();
    double delta = currentTime - lastTime;
    nbFrames++;
    if (delta >= 1.0) { // If last cout was more than 1 sec ago
        double fps = double(nbFrames) / delta;

        std::stringstream ss;
        ss << " [" << fps << " FPS]";

        glfwSetWindowTitle(pWindow, ss.str().c_str());

        nbFrames = 0;
        lastTime = currentTime;
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    WinX = width; WinY = height;
}


void processInput(GLFWwindow *window, Scene*scene)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
        sconfig->params[0] = std::min(sconfig->params[0] + 0.05f,1.0f);
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
        sconfig->params[0] = std::max(sconfig->params[0] - 0.05f,0.0f);

    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
        sconfig->params[1] = std::min(sconfig->params[1] + 0.05f,1.0f);
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS)
        sconfig->params[1] = std::max(sconfig->params[1] - 0.05f,0.0f);

    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
        sconfig->params[2] = std::min(sconfig->params[2] + 0.05f,1.0f);
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
        sconfig->params[2] = std::max(sconfig->params[2] - 0.05f,0.0f);

    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        sconfig->params[3] = std::min(sconfig->params[3] + 0.05f,1.0f);
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        sconfig->params[3] = std::max(sconfig->params[3] - 0.05f,0.0f);

    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
        sconfig->params[4] = std::min(sconfig->params[4] + 0.05f,1.0f);
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        sconfig->params[4] = std::max(sconfig->params[4] - 0.05f,0.0f);
//
//    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
//    {
//        param5 = std::min(param5 + 0.05f,1.0f);
//        translate_shape(scene,"glass0",0,0,param5*20);
//    }
//    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
//    {
//        param5 = std::max(param5 - 0.05f,0.0f);
//        translate_shape(scene,"glass0",0,0,param5*20);
//    }

    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
    {
        std::cout << sconfig->params[0]<< " "<< sconfig->params[1] << " "<< sconfig->params[2]<< " "<< sconfig->params[3]<< " "<< sconfig->params[4]<< std::endl;
    }


}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        mouseMoving = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        mouseMoving = false;
    }

}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;
    if (!mouseMoving) return;
    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

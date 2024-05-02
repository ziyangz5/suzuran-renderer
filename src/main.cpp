#include <glad/glad.h>
#include "Core.h"
#include "Camera.h"
#include "Shader.h"
#include "Scene.h"
#include <sstream>
#include <fstream>
#include "SceneParser.h"
#include "ltc.h"
#include <random>

using namespace szr;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void processInput(GLFWwindow* window);
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

    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WinX, WinY);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
    // finally check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete!" << std::endl;

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


void PostProcessingRendering(ShaderProgram* ppShader, uint& pRenderingResult)
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
    glBindTexture(GL_TEXTURE_2D, pRenderingResult);
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    ppShader->unuse();
}

auto GetExperimentTransformation(float currentFrame, int num_of_glass)
{
    glm::vec3 camera_translate(0, 0, 0);
    glm::vec2 camera_euler_delta(0, 0);
    glm::vec3 lightPositionDelta(0, 0, 0);
    std::vector<glm::vec3> glass_shift;
    for (int i = 0; i < num_of_glass; i++)
    {
        float x_delta = (i + 1) * 75;
        float z_delta = (sin(currentFrame/3.5f + (float)i / 3) + 1) / 2 * 400;
        glass_shift.push_back(glm::vec3(x_delta, 0, z_delta));
    }
    return std::make_tuple(camera_translate, camera_euler_delta, lightPositionDelta, glass_shift);

}

void render(GLFWwindow* window, Scene* scene)
{
    float lastFrame = 0.0f;

    uint ltcMatId;
    float ltc_mat[64 * 64 * 4];
    uint ltcMagId;
    float ltc_mag[64 * 64];

    uint gBuffer, pBuffer;
    //Initializing g-buffer
    glGenFramebuffers(1, &gBuffer);
    uint gNormal, gPosition, gMaterial_1, gMaterial_2, gView, gLightDir, rboDepth, screenBuffer;
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    ConfigGBuffer(gNormal, gPosition, gMaterial_1, gMaterial_2, gView, rboDepth);
    uint gAttachments[5] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_COLOR_ATTACHMENT3,GL_COLOR_ATTACHMENT4};
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //Initializing post-processing-buffer
    ShaderProgram* ppFXAA = new ShaderProgram("../shaders/pp_FXAA.glsl", ShaderProgram::eRender);
    ppFXAA->use();
    ppFXAA->setInt("screenTexture", 0);
    ppFXAA->unuse();
    glGenFramebuffers(1, &pBuffer);
    uint pRenderingResult, pDepthBuffer;
    glBindFramebuffer(GL_FRAMEBUFFER, pBuffer);
    ConfigPBuffer(pRenderingResult, pDepthBuffer);
    uint pAttachments[1] = { GL_COLOR_ATTACHMENT0 };
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

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
    int num_of_glass = 0;
    for (uint i = 0; i < scene->models.size(); i++)
    {
        if (scene->models[i]->tag.rfind("glass", 0) == 0) num_of_glass++;
    }

    glm::vec3 camera_translate; glm::vec2 camera_euler_delta; glm::vec3 lightPositionDelta; std::vector<glm::vec3> glassesPositionDelta;
    //Game loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);
        showFPS(window);

        //Geo Pass
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
        glDrawBuffers(6, gAttachments);
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
                geo_program->setVec3("lightPositionDelta", lightPositionDelta);
                Model* model = scene->models[i];
                model->Draw(camera.GetViewProjectionMatrix(WinX, WinY), camera.Position, geo_program, currentFrame);
                geo_program->unuse();
            }
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);



        //Foward Pass
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindFramebuffer(GL_FRAMEBUFFER, pBuffer);
        glDrawBuffers(1, pAttachments);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1);

        for (uint i = 0; i < scene->models.size(); i++)
        {
            ShaderProgram* program = scene->shaders[scene->models[i]->shaderID];
            if (program->name == "glass")
            {
                continue;
            }
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


        //Post-Processing Pass
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawBuffer(GL_BACK);
        PostProcessingRendering(ppFXAA,pRenderingResult);


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
    Scene* scene = SceneParser::parse_scene("../Scenes/bathroom/bathroom_opengl_area.xml");
    camera = scene->camera;
    WinX = camera.defaultWinX;
    WinY = camera.defaultWinY;
    glfwSetWindowSize(window, WinX, WinY);
    glViewport(0, 0, WinX, WinY);
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


void processInput(GLFWwindow *window)
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
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS)
    {
        requestSaveFramebuffer = 1;
    }
    if (requestSaveFramebuffer == 1 && glfwGetKey(window, GLFW_KEY_V) == GLFW_RELEASE)
    {
        requestSaveFramebuffer = 2;
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

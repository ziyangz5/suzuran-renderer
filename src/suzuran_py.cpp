#include "Core.h"
//#include "Camera.h"
//#include "Shader.h"
//#include "Scene.h"
//#include <sstream>
//#include <fstream>
#include "SceneParser.h"
#include "ltc.h"
#include <pybind11/pybind11.h>
#include <iostream>
#include "pybind11/numpy.h"
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace szr;

class Renderer
{
public:
    int width = 512;
    int height = 512;
    GLFWwindow* window;
    Renderer(const char* path_to_scene)
    {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        //glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);

        window = glfwCreateWindow(width, height, "test", NULL, NULL);
        if (window == NULL) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return;
        }
        glfwMakeContextCurrent(window);
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cout << "Failed to initialize GLAD" << std::endl;
            throw "Failed to initialize GLAD";
        }
        glViewport(0, 0, width, height);
        glEnable(GL_DEPTH_TEST);
        //glEnable(GL_CULL_FACE);
        printf("OpenGL version used by this application (%s): \n", glGetString(GL_VERSION));
        scene = szr::SceneParser::parse_scene(path_to_scene);
        width = scene->camera.defaultWinX;
        height = scene->camera.defaultWinY;
        glfwSetWindowSize(window, width, height);
        glViewport(0, 0, width, height);
        initialize_scene();
    }
    void initialize_scene()
    {
        float ltc_mat[64 * 64 * 4];
        float ltc_mag[64 * 64];

        //Initializing g-buffer
        glGenFramebuffers(1, &gBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
        ConfigGBuffer(gNormal, gPosition, gMaterial_1, gMaterial_2, gView, rboDepth);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //Initializing post-processing-buffer
        ppFXAA = new ShaderProgram("../shaders/pp_FXAA.glsl", ShaderProgram::eRender);
        ppFXAA->use();
        ppFXAA->setInt("screenTexture", 0);
        ppFXAA->unuse();
        glGenFramebuffers(1, &pBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, pBuffer);
        ConfigPBuffer(pRenderingResult, pDepthBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

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
            for (int i = 0; i < scene->lights.size(); i++)
            {
                program->setVec3("lightPositions[" + std::to_string(0 + i * 4) + "]", scene->lights[i]->pos_x1 + scene->light_position_shift[i]);
                program->setVec3("lightPositions[" + std::to_string(1 + i * 4) + "]", scene->lights[i]->pos_x2 + scene->light_position_shift[i]);
                program->setVec3("lightPositions[" + std::to_string(2 + i * 4) + "]", scene->lights[i]->pos_x3 + scene->light_position_shift[i]);
                program->setVec3("lightPositions[" + std::to_string(3 + i * 4) + "]", scene->lights[i]->pos_x4 + scene->light_position_shift[i]);
                program->setVec3("lightColors[" + std::to_string(i ) + "]", scene->lights[i]->color);
                program->setBool("twoSided", scene->lights[i]->two_sided);
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
        num_of_glass = 0;
        for (uint i = 0; i < scene->models.size(); i++)
        {
            if (scene->models[i]->tag.rfind("glass", 0) == 0) num_of_glass++;
        }
    }
    py::tuple render()
    {
	glfwMakeContextCurrent(window);
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
                Model* model = scene->models[i];
                model->Draw(scene->camera.GetViewProjectionMatrix(width, height), scene->camera.Position, geo_program, 0);
                geo_program->unuse();
            }
        }
        std::vector<float*> gbuffers = SaveGbufferHandler();
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
            model->Draw(scene->camera.GetViewProjectionMatrix(width, height), scene->camera.Position, program, 0);

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
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        float* pixels = SaveRenderedImageHandler();
        py::list glasses;
        for (int glass_id = 0; glass_id < num_of_glass; glass_id++)
        {
            //Geo Pass
            glDisable(GL_BLEND);
            glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
            glDrawBuffers(5, gAttachments);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0, 0, 0, 1);
            for (uint i = 0; i < scene->models.size(); i++)
            {
                if (scene->models[i]->tag != "glass" + std::to_string(glass_id)) continue;
                ShaderProgram* geo_program = scene->geo_shaders[scene->models[i]->shaderID];
                if (geo_program != nullptr)
                {

                    geo_program->use();
                    Model* model = scene->models[i];
                    model->Draw(scene->camera.GetViewProjectionMatrix(width, height), scene->camera.Position, geo_program, 0);
                    geo_program->unuse();
                }
            }
            std::vector<float*> gbuffers_glass = SaveGbufferHandler();
            std::vector<pybind11::array_t<float>> result;
            result.reserve(GBUFFER_SIZE+1);
            for (int i = 0;i<GBUFFER_SIZE;i++)
            {
                pybind11::capsule cleanup(gbuffers_glass[i], [](void *f) {delete[] static_cast<float*>(f);});
                result.push_back(pybind11::array_t<float>(
                        {width*height*4},          // shape
                        {sizeof(float)}, // stride
                        gbuffers_glass[i],
                        cleanup).reshape({width,height,4})
                                 );
            }
            pybind11::capsule cleanup(gbuffers_glass[GBUFFER_SIZE], [](void *f) {delete[] static_cast<float*>(f);});
            result.push_back(pybind11::array_t<float>(
                    {width*height},          // shape
                    {sizeof(float)}, // stride
                    gbuffers_glass[GBUFFER_SIZE],
                    cleanup).reshape({width,height}));

            glasses.append(py::cast(result));
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0, 0, 0, 1);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }


        glfwSwapBuffers(window);
        glfwPollEvents();

        std::vector<pybind11::array_t<float>> result;
        result.reserve(GBUFFER_SIZE+2);
        for (int i = 0;i<GBUFFER_SIZE;i++)
        {
            pybind11::capsule cleanup(gbuffers[i], [](void *f) {delete[] static_cast<float*>(f);});
            result.push_back(pybind11::array_t<float>(
                    {width*height*4},          // shape
                    {sizeof(float)}, // stride
                    gbuffers[i],
                    cleanup).reshape({width,height,4}));
        }
        {
        pybind11::capsule cleanup(gbuffers[GBUFFER_SIZE], [](void *f) {delete[] static_cast<float*>(f);});
        result.push_back(pybind11::array_t<float>(
                {width*height},          // shape
                {sizeof(float)}, // stride
                gbuffers[GBUFFER_SIZE],
                cleanup).reshape({width,height}));
        }

        pybind11::capsule cleanup(pixels, [](void *f) {delete[] static_cast<float*>(f);});
        result.push_back(pybind11::array_t<float>(
                {width*height*4},          // shape
                {sizeof(float)}, // stride
                pixels,cleanup
                ).reshape({width,height,4}));

        return py::make_tuple(py::cast(result),glasses);
    }
/*
    void set_camera_transform(float pitch, float yaw, float x, float y, float z, bool reset)
    {
        scene->camera.ProcessDirectTransform(glm::vec2(pitch,yaw),glm::vec3(x,y,z),reset);
    }
*/

    void set_camera_transform(float pos_x,float pos_y,float pos_z,
                              float tar_x,float tar_y,float tar_z,
                              float up_x,float up_y,float up_z)
    {
        glm::vec3 pos(pos_x,pos_y,pos_z);
        glm::vec3 target(tar_x,tar_y,tar_z);
        glm::vec3 up(up_x,up_y,up_z);
        glm::mat<4,4,float> to_world = glm::lookAt(pos, target, up);

        glm::mat4 inverted = glm::inverse(to_world);
        const glm::vec3 direction = -glm::vec3(inverted[2]);
        float yaw = glm::degrees(glm::atan(direction.z, direction.x));
        float pitch = glm::degrees(glm::asin(direction.y));

        scene->camera.SetTransform(glm::vec2(pitch,yaw),pos);

    }

    //TODO: Generalize the parameter changing method
    void set_variable(const char* name, const char* key, float value)
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

    void translate_shape(const char* name, float x, float y,float z)
    {
        for (uint i = 0; i < scene->models.size(); i++)
        {
            if (scene->models[i]->tag != name) continue;
            scene->models[i]->SetWorldMatrix(glm::identity<glm::mat4>());
            scene->models[i]->ApplyTransformMatrix(glm::translate(glm::vec3(x,y,z)));
            break;
        }
    }

    void set_light_position_shift(int light_id, float x, float y, float z)
    {
        scene->light_position_shift[light_id] = glm::vec3(x,y,z);
    }

    void set_light_color(int light_id, float r, float g, float b)
    {
        scene->lights[light_id]->color = glm::vec3(r,g,b);
    }

private:
    szr::Scene* scene;
    unsigned int quadVAO = 0;
    unsigned int quadVBO;
    uint gBuffer, pBuffer;
    uint gNormal, gPosition, gMaterial_1, gMaterial_2, gView, rboDepth, screenBuffer;
    static inline const unsigned int GBUFFER_SIZE = 5;
    uint gAttachments[GBUFFER_SIZE] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_COLOR_ATTACHMENT3,GL_COLOR_ATTACHMENT4};
    uint pAttachments[1] = { GL_COLOR_ATTACHMENT0 };
    ShaderProgram* ppFXAA;
    uint ltcMatId,ltcMagId;
    uint pRenderingResult, pDepthBuffer;
    int num_of_glass;

    void ConfigGBuffer(uint& gNormal, uint& gPosition, uint& gMaterial_1, uint& gMaterial_2, uint& gView, uint& rboDepth)
    {
        glGenTextures(1, &gNormal);
        glBindTexture(GL_TEXTURE_2D, gNormal);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gNormal, 0);

        glGenTextures(1, &gPosition);
        glBindTexture(GL_TEXTURE_2D, gPosition);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gPosition, 0);

        glGenTextures(1, &gMaterial_1);
        glBindTexture(GL_TEXTURE_2D, gMaterial_1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gMaterial_1, 0);

        glGenTextures(1, &gMaterial_2);
        glBindTexture(GL_TEXTURE_2D, gMaterial_2);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, gMaterial_2, 0);

        glGenTextures(1, &gView);
        glBindTexture(GL_TEXTURE_2D, gView);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, gView, 0);

        glGenRenderbuffers(1, &rboDepth);
        glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
        // finally check if framebuffer is complete
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "Framebuffer not complete!" << std::endl;

    }


    void ConfigPBuffer(uint& pRenderingResult, uint& pDepthBuffer)
    {
        glGenTextures(1, &pRenderingResult);
        glBindTexture(GL_TEXTURE_2D, pRenderingResult);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pRenderingResult, 0);

        glGenRenderbuffers(1, &pDepthBuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, pDepthBuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
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
        ppShader->setVec2("viewportSize", width, height);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, pRenderingResult);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
        ppShader->unuse();
    }

    float* SaveRenderedImageHandler()
    {
        float* pixels = new float[4 * width * height];
        std::vector<float> result;
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, pixels);
        return pixels;
    }

    std::vector<float*> SaveGbufferHandler()
    {
        std::vector<float*> result;
        for (int i = 0; i < GBUFFER_SIZE; i++)
        {
            float* pixels = new float[4 * width * height];
            glReadBuffer(gAttachments[i]);
            glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, pixels);
            result.push_back(pixels);
        }
        float* depth = new float[width * height];

        glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth);
        result.push_back(depth);
        return result;
    }
};

PYBIND11_MODULE(suzuran, handle)
{
    handle.doc() = "SuzuranRenderer v1.0.";
    //handle.def("fast_factorial", &test_factorial);
    py::class_<Renderer>(
            handle,"Renderer"
            )
            .def(py::init<const char*>())
            .def("render",[](Renderer &self) {
                return self.render();
            })
            .def("set_camera_transform",&Renderer::set_camera_transform)
            .def("set_variable",&Renderer::set_variable)
            .def("set_light_position_shift",&Renderer::set_light_position_shift)
            .def("set_light_color",&Renderer::set_light_color)
            .def("translate_shape",&Renderer::translate_shape)

            ;
}

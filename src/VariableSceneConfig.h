//
// Created by xshuj on 2/4/2024.
//

#ifndef SUZURANRENDERER_VARIABLESCENECONFIG_H
#define SUZURANRENDERER_VARIABLESCENECONFIG_H
#include <map>
#include <utility>
#include <vector>
#include <string>
#include "Core.h"
#include "Scene.h"
enum VariableType
{
    VCamera,
    VObjectPos,
    VAlpha,
    VColor,
    VLightPos
};
class VariableSceneConfig
{
    std::map<std::string,int> num_params = {{"bathroom", 6}, {"desk", 8}, {"cornellbox", 9}, {"desk", 8},{"super_cbox", 9}};
    std::map<std::string,std::vector<VariableType>> params_type = {
            {"bathroom", {VCamera,VCamera,VCamera,VCamera,VCamera,VAlpha}},
            {"desk", {VCamera,VCamera,VCamera,VCamera,VCamera,VColor,VColor,VColor}},
            {"cornellbox", {VCamera,VCamera,VCamera,VCamera,VCamera,VObjectPos,VObjectPos,VObjectPos,VLightPos}},
            {"desk", {VCamera,VCamera,VCamera,VCamera,VCamera,VColor,VColor,VColor}},
            {"super_cbox", {VCamera,VCamera,VCamera,VCamera,VCamera,VObjectPos,VObjectPos,VObjectPos,VLightPos}},
    };
    std::map<std::string,std::vector<glm::vec3>> params_range = {
            {"bathroom", {glm::vec3(0.5,0,0)}},
            {"desk", {glm::vec3(1,0,0),glm::vec3(0,1,0),glm::vec3(0,0,1)}},
            {"cornellbox", {glm::vec3(0,0,15),glm::vec3(0,0,15),glm::vec3(0,0,15),glm::vec3(20,0,0)}},
            {"desk", {glm::vec3(1,0,0),glm::vec3(0,1,0),glm::vec3(0,0,1)}},
            {"super_cbox", {glm::vec3(0,0,15),glm::vec3(0,0,15),glm::vec3(0,0,15),glm::vec3(20,0,0)}},
    };

    std::map<std::string,std::vector<glm::vec3>> params_min = {
            {"bathroom", {glm::vec3(0,0,0)}},
            {"desk", {glm::vec3(0,1,1),glm::vec3(1,0,1),glm::vec3(1,1,0)}},
            {"cornellbox", {glm::vec3(0,0,0),glm::vec3(0,0,0),glm::vec3(0,0,0),glm::vec3(-10,0,0)}},
            {"desk", {glm::vec3(0,1,1),glm::vec3(1,0,1),glm::vec3(1,1,0)}},
            {"super_cbox", {glm::vec3(0,0,0),glm::vec3(0,0,0),glm::vec3(0,0,0),glm::vec3(-10,0,0)}},
    };
    std::string scene_name;
    const float update_interval = 0.03f;
    const float speed_scale = 0.006f;
public:
    std::vector<float> params;
    VariableSceneConfig(const std::string& scene_name, szr::Scene* scene): scene_name(scene_name)
    {
        int num = num_params[scene_name];
        if (scene_name == "bathroom")
        {
            for (int i = 0;i < num; i++)
            {
                if (num < 5)
                {
                    params.push_back(0.2f);
                }
                params.push_back(0.05f);
            }
            params[2] = 0.2;
            params[3] = 0.2;
        }
        else if (scene_name == "cornellbox")
        {
            for (int i = 0;i < num; i++)
            {
                if (i < 5)
                {
                    params.push_back(1.0f);
                }
                params.push_back(0.1f);
            }
            glm::vec3 g0 = params_range[scene_name][0] * params[5] + params_min[scene_name][0];
            glm::vec3 g1 = params_range[scene_name][1] * params[6] + params_min[scene_name][1];
            glm::vec3 g2 = params_range[scene_name][2] * params[7] + params_min[scene_name][2];
            translate_shape(scene,"glass0",g0.x,g0.y,g0.z);
            translate_shape(scene,"glass1",g1.x,g1.y,g1.z);
            translate_shape(scene,"glass2",g2.x,g2.y,g2.z);
            params[0] = 0.2;
            params[1] = 0.9;
            params[2] = 0.9;
            params[3] = 1.0;
            params[4] = 0.5;

            params[8] = 0.0;

            glm::vec3 light = params_range[scene_name][3] * params[8] + params_min[scene_name][3];

            set_light_position_shift(scene,0,light.x,light.y,light.z);
        }

        else if (scene_name == "desk")
        {
            for (int i = 0;i < num; i++)
            {
                params.push_back(0.0f);
            }
            params[0] = 0.25;
            params[1] = 0.5;
            params[2] = 0.2;
            params[3] = 1.0;
            params[4] = 0.7;

            params[5] = 0.0;
            params[6] = 0.7;
            params[7] = 0.0;


        }
        else if (scene_name == "super_cbox")
        {
            for (int i = 0;i < num; i++)
            {
                if (i < 5)
                {
                    params.push_back(1.0f);
                }
                params.push_back(0.0f);
            }
            glm::vec3 g0 = params_range[scene_name][0] * params[5] + params_min[scene_name][0];
//            glm::vec3 g1 = params_range[scene_name][1] * params[6] + params_min[scene_name][1];
//            glm::vec3 g2 = params_range[scene_name][2] * params[7] + params_min[scene_name][2];
            translate_shape(scene,"glass0",g0.x,g0.y,g0.z);
//            translate_shape(scene,"glass1",g1.x,g1.y,g1.z);
//            translate_shape(scene,"glass2",g2.x,g2.y,g2.z);
            params[0] = 0.5;
            params[1] = 0.2;
            params[2] = 0.5;
            params[3] = 0.25;
            params[4] = 0.7;

            params[8] = 0.0;

            glm::vec3 light = params_range[scene_name][3] * params[8] + params_min[scene_name][3];

            set_light_position_shift(scene,0,light.x,light.y,light.z);
        }

    }

    float safe_add(float v,float delta,float max_range)
    {
        v += delta;
        if (v > max_range)
        {
            v = max_range;
        }
        return v;
    }

    float safe_sub(float v,float delta,float min_range)
    {
        v -= delta;
        if (v <= min_range)
        {
            v = min_range;
        }
        return v;
    }

    void process_transformation(szr::Scene* scene, float deltaTime)
    {
        float delta = deltaTime/update_interval*speed_scale;
        if (scene_name == "bathroom")
        {
            params[0] = safe_add(params[0],delta,0.9);
            params[1] = safe_add(params[1],delta,0.9);
            params[2] = safe_add(params[2],delta,0.9);
            params[3] = safe_add(params[3],delta,0.95);
            params[4] = safe_add(params[4],delta,0.85);

            params[5] = safe_add(params[5],delta,0.9);

            float alpha = params_range[scene_name][0].x * params[5] + params_min[scene_name][0].x;
            set_variable(scene,"glass0","roughness",alpha);
        }
        else if (scene_name == "cornellbox")
        {
            params[0] = safe_add(params[0],delta*1.2f,0.85);
            params[1] = safe_add(params[1],delta*1.2f,0.9);
            params[2] = safe_add(params[2],delta*1.2f,0.9);
            params[3] = safe_add(params[3],delta*1.2f,1.0);
            params[4] = safe_add(params[4],delta*1.2f,0.75);

            params[5] = safe_add(params[5],delta,1.0);
            params[6] = safe_add(params[6],delta,1.0);
            params[7] = safe_add(params[7],delta,1.0);

            glm::vec3 g0 = params_range[scene_name][0] * params[5] + params_min[scene_name][0];
            glm::vec3 g1 = params_range[scene_name][1] * params[6] + params_min[scene_name][1];
            glm::vec3 g2 = params_range[scene_name][2] * params[7] + params_min[scene_name][2];

            translate_shape(scene,"glass0",g0.x,g0.y,g0.z);
            translate_shape(scene,"glass1",g1.x,g1.y,g1.z);
            translate_shape(scene,"glass2",g2.x,g2.y,g2.z);

            params[8] = safe_add(params[8],delta,1.0);
            glm::vec3 light = params_range[scene_name][3] * params[8] + params_min[scene_name][3];

            set_light_position_shift(scene,0,light.x,light.y,light.z);


        }
        else if (scene_name == "desk")
        {
            params[2] = safe_add(params[2],delta*0.32f,0.5);

            params[5] = safe_add(params[5],delta*0.4f,0.5);
            params[6] = safe_sub(params[6],delta*0.4f,0.4);
            params[7] = safe_add(params[7],delta*0.4f,0.2);

            glm::vec3 g0 = params_range[scene_name][0] * params[5] + params_min[scene_name][0];
            glm::vec3 g1 = params_range[scene_name][1] * params[6] + params_min[scene_name][1];
            glm::vec3 g2 = params_range[scene_name][2] * params[7] + params_min[scene_name][2];

            set_color(scene,"glass0",g0.x,g0.y,g0.z);
            set_color(scene,"glass1",g1.x,g1.y,g1.z);
            set_color(scene,"glass2",g2.x,g2.y,g2.z);

        }
        else if (scene_name == "super_cbox")
        {
            params[0] = safe_add(params[0],delta*1.2f,0.5);
            params[1] = safe_add(params[1],delta*1.2f,0.7);
            params[2] = safe_add(params[2],delta*1.2f,0.65);
            params[3] = safe_add(params[3],delta*1.2f,0.45);
            params[4] = safe_add(params[4],delta*1.2f,0.9);

            params[5] = safe_add(params[5],delta,1.0);
            params[6] = safe_add(params[6],delta,1.0);
            params[7] = safe_add(params[7],delta,1.0);

            glm::vec3 g0 = params_range[scene_name][0] * params[5] + params_min[scene_name][0];
//            glm::vec3 g1 = params_range[scene_name][1] * params[6] + params_min[scene_name][1];
//            glm::vec3 g2 = params_range[scene_name][2] * params[7] + params_min[scene_name][2];

            translate_shape(scene,"glass0",g0.x,g0.y,g0.z);
//            translate_shape(scene,"glass1",g1.x,g1.y,g1.z);
//            translate_shape(scene,"glass2",g2.x,g2.y,g2.z);

            params[8] = safe_add(params[8],delta,1.0);
            glm::vec3 light = params_range[scene_name][3] * params[8] + params_min[scene_name][3];

            set_light_position_shift(scene,0,light.x,light.y,light.z);
        }





    }

    void translate_shape(szr::Scene* scene, const char* name, float x, float y,float z)
    {
        for (uint i = 0; i < scene->glass_models.size(); i++)
        {
            if (scene->glass_models[i]->tag != name) continue;
            scene->glass_models[i]->SetWorldMatrix(glm::identity<glm::mat4>());
            scene->glass_models[i]->ApplyTransformMatrix(glm::translate(glm::vec3(x,y,z)));
            break;
        }
    }

    void set_variable(szr::Scene* scene,const std::string& name, const char* key, float value)
    {
        for (uint i = 0; i < scene->glass_models.size(); i++)
        {
            if (scene->glass_models[i]->tag != name) continue;
            szr::ShaderProgram* geo_program = scene->geo_shaders[scene->glass_models[i]->shaderID];

            if (geo_program != nullptr)
            {

                geo_program->use();
                geo_program->setFloat(key,value);
                geo_program->unuse();
            }
        }
    }

    void set_color(szr::Scene* scene,const char* name, float r, float g, float b)
    {
        for (uint i = 0; i < scene->glass_models.size(); i++)
        {
            if (scene->glass_models[i]->tag != name) continue;
            szr::ShaderProgram* geo_program = scene->geo_shaders[scene->glass_models[i]->shaderID];
            if (geo_program != nullptr)
            {
                geo_program->use();
                geo_program->setVec3("reflectance",r,g,b);
                geo_program->unuse();
            }
        }
    }

    void set_light_position_shift(szr::Scene* scene,int light_id, float x, float y, float z)
    {
        scene->light_position_shift[light_id] = glm::vec3(x,y,z);
    }


};

#endif //SUZURANRENDERER_VARIABLESCENECONFIG_H

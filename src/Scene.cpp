#include "Scene.h"
using namespace szr;

Scene::Scene(std::vector<Model*> models,
             std::vector<Model*> gmodels,
             std::vector<Light*> lights,
             std::vector<ShaderProgram*> shaders,
             std::vector<ShaderProgram*> geo_shaders,
             Camera cam):models(models), lights(lights), shaders(shaders), geo_shaders(geo_shaders), camera(cam),glass_models(gmodels)
{
    for (int i = 0; i<lights.size();i++)
    {
        light_position_shift.emplace_back(0,0,0);
    }
};

Scene::~Scene()
{

};
//
// Created by nxsnow on 5/14/23.
//

#ifndef SUZURANRENDERER_SCENEPARSER_H
#define SUZURANRENDERER_SCENEPARSER_H
#include "Scene.h"
#include "3rdparty/pugixml.hpp"
#include <regex>
#include <filesystem>

namespace szr
{

    class SceneParser
    {
    private:
        static inline std::filesystem::path old_path;
        static inline std::filesystem::path current_path;
    public:
        static std::vector<std::string> split_string(const std::string& str, const std::regex& delim_regex)
        {
            std::sregex_token_iterator first{ begin(str), end(str), delim_regex, -1 }, last;
            std::vector<std::string> list{ first, last };
            return list;
        }

        static bool parse_boolean(const std::string& value)
        {
            if (value == "true")
            {
                return true;
            }
            return false;
        }

        static glm::mat4 parse_mat4(const std::string& value)
        {
            std::vector<std::string> list = split_string(value, std::regex("(,| )+"));
            if (list.size() != 16) {
                throw ("parse_mat4 failed");
            }

            glm::mat4 m;
            int k = 0;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    m[i][j] = std::stof(list[k++]);
                }
            }
            return m;
        }
        static glm::vec3 parse_vector3(const std::string& value)
        {
            std::vector<std::string> list = split_string(value, std::regex("(,| )+"));
            glm::vec3 v;
            if (list.size() == 1) {
                v[0] = std::stof(list[0]);
                v[1] = std::stof(list[0]);
                v[2] = std::stof(list[0]);
            }
            else if (list.size() == 3) {
                v[0] = std::stof(list[0]);
                v[1] = std::stof(list[1]);
                v[2] = std::stof(list[2]);
            }
            else {
                throw ("parse_vector3 failed");
            }
            return v;
        }


        static glm::mat4 parse_transform(pugi::xml_node node)
        {
            glm::mat4 tform = glm::mat4(1);
            for (auto child : node.children()) {
                std::string name = child.name();
                std::transform(name.begin(), name.end(), name.begin(), ::tolower);

                if (name == "scale") {
                    float x = 1.0;
                    float y = 1.0;
                    float z = 1.0;
                    if (!child.attribute("x").empty())
                        x = std::stof(child.attribute("x").value());
                    if (!child.attribute("y").empty())
                        y = std::stof(child.attribute("y").value());
                    if (!child.attribute("z").empty())
                        z = std::stof(child.attribute("z").value());
                    tform = glm::scale(glm::vec3(x, y, z)) * tform;
                }
                else if (name == "translate") {
                    float x = 0.0;
                    float y = 0.0;
                    float z = 0.0;
                    if (!child.attribute("x").empty())
                        x = std::stof(child.attribute("x").value());
                    if (!child.attribute("y").empty())
                        y = std::stof(child.attribute("y").value());
                    if (!child.attribute("z").empty())
                        z = std::stof(child.attribute("z").value());
                    tform = glm::translate(glm::vec3(x, y, z)) * tform;
                }
                else if (name == "rotate") {
                    float x = 0.0;
                    float y = 0.0;
                    float z = 0.0;
                    float angle = 0.0;
                    if (!child.attribute("x").empty())
                        x = std::stof(child.attribute("x").value());
                    if (!child.attribute("y").empty())
                        y = std::stof(child.attribute("y").value());
                    if (!child.attribute("z").empty())
                        z = std::stof(child.attribute("z").value());
                    if (!child.attribute("angle").empty())
                        angle = std::stof(child.attribute("angle").value());
                    tform = glm::rotate(angle * 180.0f / 3.141592653f, glm::vec3(x, y, z)) * tform;
                }
                else if (name == "lookat") {
                    glm::vec3 pos = parse_vector3(child.attribute("origin").value());
                    glm::vec3 target = parse_vector3(child.attribute("target").value());
                    glm::vec3 up = parse_vector3(child.attribute("up").value());
                    tform = glm::lookAt(pos, target, up) * tform;
                }
                else if (name == "matrix") {
                    glm::mat4 trans = parse_mat4(std::string(child.attribute("value").value()));
                    tform = trans * tform;
                }
            }
            return tform;
        }

        static glm::ivec2 parse_screen_size(pugi::xml_node node)
        {
            glm::ivec2 size(800, 800);
            for (auto child : node.children()) {
                std::string name = child.attribute("name").value();
                if (name == "width") {
                    size.x = std::stoi(child.attribute("value").value());
                }
                else if (name == "height") {
                    size.y = std::stoi(child.attribute("value").value());
                }
            }
            return size;
        }

        static Camera parse_camera(pugi::xml_node node)
        {
            float fov = 45.0f;
            glm::mat4 to_world(1);
            glm::ivec2 size(512, 512);
            std::string type = node.attribute("type").value();
            if (type == "perspective") {
                for (auto child : node.children()) {
                    std::string name = child.attribute("name").value();
                    if (name == "fov") {
                        fov = std::stof(child.attribute("value").value());
                    }
                    else if (name == "toWorld"||name == "to_world") {
                        to_world = parse_transform(child);
                    }
                    else if (name == "fovAxis") {
                        //No need in opengl
                    }
                    std::string type = child.attribute("type").value();
                    if (type == "hdrfilm")
                    {
                        size = parse_screen_size(child);
                    }
                }
            }
            else {
                throw (std::string("Unsupported sensor: ") + type);
            }

            glm::mat4 inverted = glm::inverse(to_world);
            glm::vec3 pos = glm::vec3(inverted[3]);
            const glm::vec3 direction = -glm::vec3(inverted[2]);
            float yaw = glm::degrees(glm::atan(direction.z, direction.x));
            float pitch = glm::degrees(glm::asin(direction.y));
            Camera cam(pos, glm::vec3(0, 1, 0), yaw, pitch, fov);
            cam.defaultWinX = size.x;
            cam.defaultWinY = size.y;
            return cam;
        }

        static Texture* parse_texture(pugi::xml_node node);


        static auto parse_bsdf(pugi::xml_node node, std::string& material_name)
        {
            std::string type = node.attribute("type").value();
            std::string id;
            if (!node.attribute("id").empty()) {
                id = node.attribute("id").value();
            }
            ShaderProgram* geo_s = nullptr;
            ShaderProgram* s = new ShaderProgram(("../shaders/" +type + ".glsl").c_str(), ShaderProgram::eRender);

            if (std::filesystem::exists(("../shaders/" +type + "_geo.glsl").c_str()))
            {
                geo_s = new ShaderProgram(("../shaders/" +type + "_geo.glsl").c_str(), ShaderProgram::eRender);
                geo_s->name = type + "_geo";
            }

            s->name = type;
            material_name = id;
            if (type == "diffuse")
            {
                for (auto child : node.children())
                {
                    glm::vec3 reflectance;
                    std::string name = child.attribute("name").value();
                    if (name == "reflectance") {
                        reflectance = parse_vector3(child.attribute("value").value());
                        s->use();
                        s->setVec3(name.c_str(), reflectance);
                    }
                    else if (name == "texture")
                    {
                        std::cout << "Implementing.." << std::endl;
                    }
                }
                s->unuse();
                return std::tie(s, geo_s);
            }
            else if (type == "gl_ggx")
            {
                for (auto child : node.children())
                {
                    glm::vec3 albedo;
                    float metallic; float roughness; float ao;
                    std::string name = child.attribute("name").value();
                    if (name == "albedo")
                    {
                        albedo = parse_vector3(child.attribute("value").value());
                        s->setVec3(name.c_str(), albedo);
                    }
                    if (name == "metallic")
                    {
                        auto test = child.attribute("value").value();
                        metallic = std::stof(test);
                        s->setFloat(name.c_str(), metallic);
                    }
                    if (name == "roughness")
                    {
                        roughness = std::stof(child.attribute("value").value());
                        s->setFloat(name.c_str(), roughness);
                    }
                    if (name == "ao")
                    {
                        ao = std::stof(child.attribute("value").value());
                        s->setFloat(name.c_str(), ao);
                    }
                    else if (name == "texture")
                    {
                        std::cout << "Implementing.." << std::endl;
                    }
                }
                return std::tie(s, geo_s);
            }
            else if (type == "area_ggx")
            {
                for (auto child : node.children())
                {
                    glm::vec3 reflectance;
                    float roughness = 0.01;
                    std::string name = child.attribute("name").value();
                    std::string attribute_name = child.name();
                    if (name == "reflectance") {
                        reflectance = parse_vector3(child.attribute("value").value());
                        s->use();
                        s->setVec3(name.c_str(), reflectance);
                        if (geo_s != nullptr)
                        {
                            geo_s->use();
                            geo_s->setVec3(name.c_str(), reflectance);
                        }
                    }
                    else if (name == "roughness")
                    {
                        roughness = std::stof(child.attribute("value").value());
                        s->use();
                        s->setFloat(name.c_str(), roughness);
                        if (geo_s != nullptr)
                        {
                            geo_s->use();
                            geo_s->setFloat(name.c_str(), roughness);
                        }
                    }
                    else if (attribute_name == "texture")
                    {
                        Texture* texture = parse_texture(child);
                        s->use();
                        s->setTexture(texture);
                        s->use();
                        s->setTexture(texture);
                        if (geo_s != nullptr)
                        {
                            geo_s->use();
                            geo_s->setTexture(texture);
                        }
                    }
                }
                s->unuse();
                return std::tie(s, geo_s);
            }
            else if (type == "glass")
            {
                for (auto child : node.children())
                {
                    glm::vec3 reflectance;
                    float roughness = 0.01;
                    std::string name = child.attribute("name").value();
                    if (name == "reflectance") {
                        reflectance = parse_vector3(child.attribute("value").value());
                        s->use();
                        s->setVec3(name.c_str(), reflectance);
                        if (geo_s != nullptr)
                        {
                            geo_s->use();
                            geo_s->setVec3(name.c_str(), reflectance);
                        }
                    }
                }
                return std::tie(s, geo_s);
            }
            else if (true)//TODO: Add more bsdf
            {
                return std::tie(s, geo_s);
            }
        }

        static Model* parse_shape(pugi::xml_node node, std::vector<ShaderProgram*>& shaders, std::vector<ShaderProgram*>& geo_shaders, std::map<std::string, int>& shader_map)
        {
            int material_id = -1;
            for (auto child : node.children()) {
                std::string name = child.name();
                if (name == "ref") {
                    std::string name_value = child.attribute("name").value();
                    pugi::xml_attribute id = child.attribute("id");
                    if (id.empty()) {
                        throw ("Material reference id not specified.");
                    }
                    if (name_value == "interior") {
                        throw ("Medium not supported");
                    }
                    else if (name_value == "exterior") {
                        throw ("Medium not supported");
                    }
                    else {
                        auto it = shader_map.find(id.value());
                        if (it == shader_map.end()) {
                            throw (std::string("Material reference ") + id.value() + std::string(" not found."));
                        }
                        material_id = it->second;
                    }
                }
                else if (name == "bsdf") {
                    std::string material_name;
                    ShaderProgram* s; ShaderProgram* geo_s;
                    std::tie(s, geo_s) = parse_bsdf(child, material_name);
                    if (!material_name.empty()) {
                        shader_map[material_name] = shaders.size();
                    }
                    material_id = shaders.size();
                    shaders.push_back(s);
                    geo_shaders.push_back(geo_s);
                }
                else if (name == "medium") {
                    throw ("Medium not supported");
                }
            }

            Model* shape = new Model();
            std::string type = node.attribute("type").value();
            if (type == "obj") {
                std::string filename;
                glm::mat4 to_world(1);
                for (auto child : node.children()) {
                    std::string name = child.attribute("name").value();
                    if (name == "filename") {
                        filename = child.attribute("value").value();
                    }
                    else if (name == "toWorld") {
                        if (std::string(child.name()) == "transform") {
                            to_world = parse_transform(child);
                        }
                    }
                }
                shape->LoadObj(filename);
                shape->SetWorldMatrix(to_world);
                shape->shaderID = material_id;
            }
            else
            {
                throw ("Only supports .obj for now");
            }

            std::string tag;
            if (!node.attribute("tag").empty()) {
                tag = node.attribute("tag").value();
            }
            shape->tag = tag;
            return shape;
        }

        static Light* parse_light(pugi::xml_node node)
        {
            std::string type = node.attribute("type").value();
            Light* l = new Light();
            if (type == "point")
            {
                l->type = LightType::Point;
                for (auto child : node.children())
                {
                    std::string name = child.name();
                    if (name == "position")
                    {
                        float x = 0.0;
                        float y = 0.0;
                        float z = 0.0;
                        if (!child.attribute("x").empty())
                            x = std::stof(child.attribute("x").value());
                        if (!child.attribute("y").empty())
                            y = std::stof(child.attribute("y").value());
                        if (!child.attribute("z").empty())
                            z = std::stof(child.attribute("z").value());
                        l->position = glm::vec3(x, y, z);
                    }
                    if (name == "color")
                    {
                        glm::vec3 color = parse_vector3(child.attribute("value").value());
                        l->color = color;
                    }
                }
            }
            else if (type == "area")
            {
                l->type = LightType::Area;
                for (auto child : node.children())
                {
                    std::string name = child.name();
                    if (name == "x1")
                    {
                        float x = 0.0;
                        float y = 0.0;
                        float z = 0.0;
                        if (!child.attribute("x").empty())
                            x = std::stof(child.attribute("x").value());
                        if (!child.attribute("y").empty())
                            y = std::stof(child.attribute("y").value());
                        if (!child.attribute("z").empty())
                            z = std::stof(child.attribute("z").value());
                        l->pos_x1 = glm::vec3(x, y, z);
                    }
                    else if (name == "x2")
                    {
                        float x = 0.0;
                        float y = 0.0;
                        float z = 0.0;
                        if (!child.attribute("x").empty())
                            x = std::stof(child.attribute("x").value());
                        if (!child.attribute("y").empty())
                            y = std::stof(child.attribute("y").value());
                        if (!child.attribute("z").empty())
                            z = std::stof(child.attribute("z").value());
                        l->pos_x2 = glm::vec3(x, y, z);
                    }
                    else if (name == "x3")
                    {
                        float x = 0.0;
                        float y = 0.0;
                        float z = 0.0;
                        if (!child.attribute("x").empty())
                            x = std::stof(child.attribute("x").value());
                        if (!child.attribute("y").empty())
                            y = std::stof(child.attribute("y").value());
                        if (!child.attribute("z").empty())
                            z = std::stof(child.attribute("z").value());
                        l->pos_x3 = glm::vec3(x, y, z);
                    }
                    else if (name == "x4")
                    {
                        float x = 0.0;
                        float y = 0.0;
                        float z = 0.0;
                        if (!child.attribute("x").empty())
                            x = std::stof(child.attribute("x").value());
                        if (!child.attribute("y").empty())
                            y = std::stof(child.attribute("y").value());
                        if (!child.attribute("z").empty())
                            z = std::stof(child.attribute("z").value());
                        l->pos_x4 = glm::vec3(x, y, z);
                    }
                    if (name == "color")
                    {
                        glm::vec3 color = parse_vector3(child.attribute("value").value());
                        l->color = color;
                    }
                    if (name == "two_sided")
                    {
                        bool is_two_sided = parse_boolean(child.attribute("value").value());
                        l->two_sided = is_two_sided;
                    }
                }
            }
            else
            {
                std::cerr << "Not supported yet: " + type + " light" << std::endl;
            }
            return l;
        }

        static Scene* parse_scene(std::filesystem::path path)
        {

            pugi::xml_document doc;
            pugi::xml_parse_result result = doc.load_file(path.c_str());
            if (!result) {
                std::cerr << "Error description: " << result.description() << std::endl;
                std::cerr << "Error offset: " << result.offset << std::endl;
                throw ("Parse error");
            }
            old_path = std::filesystem::current_path();
            current_path = path.parent_path();
            std::filesystem::current_path(current_path);

            pugi::xml_node node = doc.child("scene");

            std::vector<Model*> models;
            std::vector<Model*> glass_models;
            std::vector<Light*> lights;
            std::vector<ShaderProgram*> shaders;
            std::vector<ShaderProgram*> geo_shaders;
            std::map<std::string /* name id */, int /* index id */> shader_map;
            Camera cam;
            for (auto child : node.children()) {
                std::string name = child.name();
                if (name == "integrator") {
                    //No need in opengl
                    continue;
                }
                else if (name == "sensor") {
                    cam = parse_camera(child);
                }
                else if (name == "bsdf") {
                    std::filesystem::current_path(old_path);

                    std::string material_name;
                    ShaderProgram* s; ShaderProgram* geo_s;
                    std::tie(s, geo_s) = parse_bsdf(child, material_name);
                    if (!material_name.empty()) {
                        shader_map[material_name] = shaders.size();
                        shaders.push_back(s);
                        geo_shaders.push_back(geo_s);
                    }

                    std::filesystem::current_path(current_path);
                }
                else if (name == "shape") {
                    Model* m = parse_shape(child, shaders, geo_shaders, shader_map);

                    if (m->tag.rfind("glass", 0) == 0)
                    {
                        glass_models.push_back(m);
                    }
                    else
                    {
                        models.push_back(m);
                    }
                }
                else if (name == "texture") {
                    //TODO: Implement texture
                }
                else if (name == "emitter") {
                    Light* l = parse_light(child);
                    lights.push_back(l);
                }
                else if (name == "medium") {
                    //No need in opengl
                }
                else
                {
                    std::cerr << "Unknown key: "+name << std::endl;
                }
            }
            std::filesystem::current_path(old_path);
            return new Scene(models,glass_models, lights, shaders, geo_shaders, cam);
        }



    };

}


#endif //SUZURANRENDERER_SCENEPARSER_H

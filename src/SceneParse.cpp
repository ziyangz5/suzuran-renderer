//
// Created by nxsnow on 5/16/23.
//
#include "SceneParser.h"
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "3rdparty/stb_image.h"
#endif

szr::Texture* szr::SceneParser::parse_texture(pugi::xml_node node)
{
    std::filesystem::current_path(current_path);
    Texture* texture = nullptr;
    for (auto child : node.children()) {
        std::string name = child.attribute("name").value();

        if (name == "filename")
        {
            int width, height, nrChannels;
            std::string filename = child.attribute("value").value();
            stbi_set_flip_vertically_on_load(true);
            unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrChannels, 0);
            texture = new Texture(data, width, height, nrChannels);
            if (!data)
            {
                std::cerr << "Unable to open texture:" + filename << std::endl;
            }

        }
    }
    std::filesystem::current_path(old_path);
    return texture;
}
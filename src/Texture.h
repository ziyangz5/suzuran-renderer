//
// Created by nxsnow on 5/16/23.
//

#ifndef SUZURANRENDERER_TEXTURE_H
#define SUZURANRENDERER_TEXTURE_H
namespace szr
{
    class Texture
    {
    public:
        unsigned char* image_data;
        int width;
        int height;
        int n_channels;
        Texture(unsigned char* image_data, int width, int height, int n_channels)
                :image_data(image_data), width(width), height(height), n_channels(n_channels)
        {

        }

    };
}
#endif //SUZURANRENDERER_TEXTURE_H

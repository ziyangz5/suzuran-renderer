//
// Created by nxsnow on 5/14/23.
//

#ifndef SUZURANRENDERER_MODEL_H
#define SUZURANRENDERER_MODEL_H
#include "Core.h"
#include "Shader.h"

namespace szr
{

    struct ModelVertex {
        glm::vec3 Position;
        glm::vec3 Normal;
        glm::vec2 TexCoord;
    };


    ////////////////////////////////////////////////////////////////////////////////

    class Model {
    public:
        Model();
        ~Model();

        void Draw(const glm::mat4& viewProjMtx, const glm::vec3& camPos, const ShaderProgram* shader, const float& time);

        void LoadObj(const std::string& path);
        void SetBuffers(const std::vector<ModelVertex>& vtx, const std::vector<uint>& idx);
        void SetWorldMatrix(glm::mat4 worldMtx);
        void ApplyTransformMatrix(glm::mat4 tMtx);
        // Access functions
        int shaderID;
        std::string tag;
    private:
        uint VertexBuffer;
        uint IndexBuffer;

        uint vao;
        int Count;
        int vertexCount;
        glm::mat4x4 modelMtx;

    };

}

#endif //SUZURANRENDERER_MODEL_H

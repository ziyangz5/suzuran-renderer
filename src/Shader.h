//
// Created by nxsnow on 5/14/23.
//

#ifndef SUZURANRENDERER_SHADER_H
#define SUZURANRENDERER_SHADER_H
#include "Core.h"
#include "Texture.h"
////////////////////////////////////////////////////////////////////////////////

namespace szr
{

    class Shader {
    public:
        enum ShaderType { eGeometry, eVertex, eFragment, eCompute };

        Shader(const char* filename, ShaderType type);
        ~Shader();

        // Access functions
        uint GetShaderID() { return ShaderID; }

    private:
        uint ShaderID;
    };

    ////////////////////////////////////////////////////////////////////////////////

    class ShaderProgram {
    public:
        enum ProgramType { eGeometry, eRender, eCompute };

        ShaderProgram(const char* filename, ProgramType type);
        ~ShaderProgram();

        // Access functions
        uint GetProgramID() const { return ProgramID; }

        void use() const
        {
            glUseProgram(ProgramID);
        }

        void unuse() const
        {
            glUseProgram(0);
        }

        // utility uniform functions
        // ------------------------------------------------------------------------
        void setBool(const std::string& name, bool value) const
        {
            glUniform1i(glGetUniformLocation(ProgramID, name.c_str()), (int)value);
        }
        // ------------------------------------------------------------------------
        void setInt(const std::string& name, int value) const
        {
            glUniform1i(glGetUniformLocation(ProgramID, name.c_str()), value);
        }
        // ------------------------------------------------------------------------
        void setFloat(const std::string& name, float value) const
        {
            glUniform1f(glGetUniformLocation(ProgramID, name.c_str()), value);
        }
        // ------------------------------------------------------------------------
        void setVec2(const std::string& name, const glm::vec2& value) const
        {
            glUniform2fv(glGetUniformLocation(ProgramID, name.c_str()), 1, &value[0]);
        }
        void setVec2(const std::string& name, float x, float y) const
        {
            glUniform2f(glGetUniformLocation(ProgramID, name.c_str()), x, y);
        }
        // ------------------------------------------------------------------------
        void setVec3(const std::string& name, const glm::vec3& value) const
        {
            glUniform3fv(glGetUniformLocation(ProgramID, name.c_str()), 1, &value[0]);
        }
        void setVec3(const std::string& name, float x, float y, float z) const
        {
            glUniform3f(glGetUniformLocation(ProgramID, name.c_str()), x, y, z);
        }
        // ------------------------------------------------------------------------
        void setVec4(const std::string& name, const glm::vec4& value) const
        {
            glUniform4fv(glGetUniformLocation(ProgramID, name.c_str()), 1, &value[0]);
        }
        void setVec4(const std::string& name, float x, float y, float z, float w)
        {
            glUniform4f(glGetUniformLocation(ProgramID, name.c_str()), x, y, z, w);
        }
        // ------------------------------------------------------------------------
        void setMat2(const std::string& name, const glm::mat2& mat) const
        {
            glUniformMatrix2fv(glGetUniformLocation(ProgramID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
        }
        // ------------------------------------------------------------------------
        void setMat3(const std::string& name, const glm::mat3& mat) const
        {
            glUniformMatrix3fv(glGetUniformLocation(ProgramID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
        }
        // ------------------------------------------------------------------------
        void setMat4(const std::string& name, const glm::mat4& mat) const
        {
            glUniformMatrix4fv(glGetUniformLocation(ProgramID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
        }

        void setTexture(Texture* texture)
        {
            unsigned int texture_id;
            glGenTextures(1, &texture_id);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            // set the texture wrapping/filtering options (on the currently bound texture object)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture->width, texture->height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, texture->image_data);
            glGenerateMipmap(GL_TEXTURE_2D);
            this->texture = texture_id;
            has_texture = true;
            setBool("textured", true);

        }

        unsigned int getTexture()
        {
            return texture;
        }
        std::string name;
        bool has_texture = false;
    private:
        enum { eMaxShaders = 3 };
        ProgramType Type;
        Shader* Shaders[eMaxShaders];
        unsigned int texture;
        uint ProgramID;
    };

    ////////////////////////////////////////////////////////////////////////////////

}


#endif //SUZURANRENDERER_SHADER_H

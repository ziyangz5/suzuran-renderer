#include "Model.h"
#include "3rdparty/fast_obj.h"

////////////////////////////////////////////////////////////////////////////////
using namespace szr;

Model::Model() {
    glGenBuffers(1, &VertexBuffer);
    glGenBuffers(1, &IndexBuffer);
    glGenVertexArrays(1, &vao);
    Count = 0; vertexCount = 0;
}

////////////////////////////////////////////////////////////////////////////////

Model::~Model() {
    glDeleteBuffers(1, &IndexBuffer);
    glDeleteBuffers(1, &VertexBuffer);
}


////////////////////////////////////////////////////////////////////////////////

void Model::Draw(const glm::mat4& viewProjMtx, const glm::vec3& camPos, const ShaderProgram* shader, const float& time) {
    // Set up shader
    shader->use();
    shader->setFloat("uTime", time);
    shader->setMat4("ModelMtx", modelMtx);

    glm::mat4 mvpMtx = viewProjMtx * modelMtx;
    shader->setMat4("ModelViewProjMtx", mvpMtx);

    glm::mat4 invTransModelMtx = glm::transpose(glm::inverse(modelMtx));
    shader->setMat4("InvTransModelMtx", invTransModelMtx);
    shader->setVec3("camPos", camPos);

    // Set up state
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBuffer);
    uint posLoc = 0;
    glEnableVertexAttribArray(posLoc);
    glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, sizeof(ModelVertex), 0);

    uint normLoc = 1;
    glEnableVertexAttribArray(normLoc);
    glVertexAttribPointer(normLoc, 3, GL_FLOAT, GL_FALSE, sizeof(ModelVertex), (void*)12);

    uint texCoordLoc = 2;
    glEnableVertexAttribArray(texCoordLoc);
    glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, GL_FALSE, sizeof(ModelVertex), (void*)24);


    // Draw geometry
    glDrawElements(GL_TRIANGLES, Count, GL_UNSIGNED_INT, 0);

    // Clean up state
    glDisableVertexAttribArray(normLoc);
    glDisableVertexAttribArray(posLoc);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    shader->unuse();
}

////////////////////////////////////////////////////////////////////////////////

void Model::LoadObj(const std::string& path)
{
    std::vector<ModelVertex> vtx2; std::vector<uint> idx2;
    fastObjMesh* mesh = fast_obj_read(path.c_str());
    for (int i = 0; i < mesh->index_count; i++)
    {
        auto mi = mesh->indices[i];
        glm::vec3 pos(mesh->positions[3 * mi.p], mesh->positions[3 * mi.p + 1], mesh->positions[3 * mi.p + 2]);
        glm::vec3 norm(mesh->normals[3 * mi.n], mesh->normals[3 * mi.n + 1], mesh->normals[3 * mi.n + 2]);
        glm::vec2 tex_coord(0, 0);
        if (mesh->texcoord_count > 1)
        {
            tex_coord = glm::vec2(mesh->texcoords[2 * mi.t], mesh->texcoords[2 * mi.t + 1]);
        }

        vtx2.push_back(ModelVertex{ pos,norm,tex_coord });
        idx2.push_back(vtx2.size() - 1);
    }

    // Create vertex & index buffers
    SetBuffers(vtx2, idx2);
}

////////////////////////////////////////////////////////////////////////////////

void Model::SetBuffers(const std::vector<ModelVertex>& vtx, const std::vector<uint>& idx) {
    Count = (int)idx.size();
    vertexCount = (int)vtx.size();
    glBindVertexArray(vao);
    // Store vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(ModelVertex), &vtx[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Store index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint), &idx[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Model::SetWorldMatrix(glm::mat4 worldMtx)
{
    this->modelMtx = worldMtx;
}

void Model::ApplyTransformMatrix(glm::mat4 tMtx)
{
    this->modelMtx = tMtx*this->modelMtx;
}
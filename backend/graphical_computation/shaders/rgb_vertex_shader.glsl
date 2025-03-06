#version 410 core

uniform mat4 camera;

in vec3 in_vertex;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    v_uv = in_uv;

    gl_Position = camera * vec4(in_vertex, 1.0);
}
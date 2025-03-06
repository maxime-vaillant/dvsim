#version 410 core

uniform sampler2D Texture;

in vec2 v_uv;

layout (location = 0) out vec4 out_color;

void main() {
    out_color = texture(Texture, v_uv);
}
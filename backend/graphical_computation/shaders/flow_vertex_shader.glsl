#version 410 core

in vec3 in_vertex;

uniform mat4 prev_camera;
uniform mat4 cur_camera;

out vec4 prev_clip;
out vec4 cur_clip;

void main() {
    // Transform positions into clip space
    prev_clip = prev_camera * vec4(in_vertex, 1.0);
    cur_clip = cur_camera * vec4(in_vertex, 1.0);

    // Output the current frame position for rendering
    gl_Position = cur_clip;
}
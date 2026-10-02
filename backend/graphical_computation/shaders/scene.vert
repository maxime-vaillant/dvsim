#version 410 core

// mvp = projection * view * model, composed in float64 on the CPU. The renderer y-flips the projection so that
// GL row 0 is the image's top row: NDC +y points DOWN the image.
uniform mat4 cur_mvp;
uniform mat4 next_mvp;

in vec3 in_vertex;
in vec2 in_uv;      // only bound for textured objects
in vec4 in_color;   // only bound for vertex-coloured objects

out vec2 v_uv;
out vec4 v_color;
out vec4 v_cur;
out vec4 v_next;

void main() {
    vec4 p = vec4(in_vertex, 1.0);
    v_cur = cur_mvp * p;
    v_next = next_mvp * p;
    v_uv = in_uv;
    v_color = in_color;
    gl_Position = v_cur;
}

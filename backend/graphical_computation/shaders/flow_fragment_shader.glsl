#version 410 core

in vec4 prev_clip;
in vec4 cur_clip;

uniform vec2 screen_size; // (width, height) of the framebuffer

out vec4 frag_color;

void main() {
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
    vec2 cur_ndc = cur_clip.xy / cur_clip.w;

    // Compute pixel displacement
    vec2 pixel_motion = (cur_ndc - prev_ndc) * screen_size;

    // Store raw pixel displacement (no encoding needed)
    frag_color = vec4(pixel_motion, 0.0, 1.0);
}
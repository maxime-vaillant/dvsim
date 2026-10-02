#version 410 core

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);  // Rec.709, applied to LINEAR light
const float W_EPS = 1e-6;                         // target must lie in front of the next camera

uniform sampler2D tex;      // sRGB8 texture: filtering and mipmapping happen in linear light
uniform int color_mode;     // 0 = base_color, 1 = texture, 2 = vertex colours
uniform vec4 base_color;    // sRGB-encoded
uniform vec2 screen_size;   // (width, height) in pixels

in vec2 v_uv;
in vec4 v_color;
in vec4 v_cur;
in vec4 v_next;

layout(location = 0) out vec4 o_rgb;        // RGBA8, sRGB-encoded, RGB order
layout(location = 1) out float o_lum;       // linear luminance
layout(location = 2) out vec2 o_flow;       // pixels, (dx right, dy down), current -> next camera
layout(location = 3) out float o_depth;     // z-depth of the current camera (= clip w)
layout(location = 4) out float o_valid;     // 1 where the flow target is in front of the next camera

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

void main() {
    vec3 lin;
    if (color_mode == 1) {
        lin = texture(tex, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;  // v = 0 is the bottom row of the image
    } else if (color_mode == 2) {
        lin = srgb_to_linear(v_color.rgb);
    } else {
        lin = srgb_to_linear(base_color.rgb);
    }
    o_rgb = vec4(linear_to_srgb(lin), 1.0);
    o_lum = dot(lin, LUMA);

    o_depth = v_cur.w;
    bool ok = v_next.w > W_EPS;
    vec2 ndc_cur = v_cur.xy / v_cur.w;
    vec2 ndc_next = ok ? v_next.xy / v_next.w : ndc_cur;
    o_flow = (ndc_next - ndc_cur) * 0.5 * screen_size;  // NDC range is 2 -> half; +y already points down
    o_valid = ok ? 1.0 : 0.0;
}

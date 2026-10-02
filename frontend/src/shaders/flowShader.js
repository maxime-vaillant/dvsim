import * as THREE from "three";

import { COLOR_WHEEL } from "../utils/flowColors.js";

// Scene override material: per-pixel camera-motion flow between the previous and the current view camera pose,
// coloured like the simulator's flow videos (see utils/flowColors.js; GLSL mirror of flowToRgb).
// Flow is expressed in pixels of the SIMULATED image (uSimSize) per simulation step (uStepRatio), so the
// colour scale means the same thing as the simulator's `flow_max_px`.

export function createFlowMaterial() {
    return new THREE.ShaderMaterial({
        uniforms: {
            uPrevViewProj: { value: new THREE.Matrix4() },
            uSimSize: { value: new THREE.Vector2(346, 260) },
            uStepRatio: { value: 1 },
            uScale: { value: 10 },
            uWheel: { value: COLOR_WHEEL.map(([r, g, b]) => new THREE.Vector3(r / 255, g / 255, b / 255)) }
        },

        vertexShader: /* glsl */`
            uniform mat4 uPrevViewProj;

            varying vec4 vCur;
            varying vec4 vPrev;

            void main() {
                vec4 world = modelMatrix * vec4( position, 1.0 );

                vCur = projectionMatrix * viewMatrix * world;
                vPrev = uPrevViewProj * world;
                gl_Position = vCur;
            }`,

        fragmentShader: /* glsl */`
            #define NCOLS ${COLOR_WHEEL.length}

            uniform vec2 uSimSize;
            uniform float uStepRatio;
            uniform float uScale;
            uniform vec3 uWheel[NCOLS];

            varying vec4 vCur;
            varying vec4 vPrev;

            void main() {
                // the surface point was behind the previous camera: no meaningful flow (black, like invalid pixels)
                if ( vPrev.w <= 1e-6 ) {
                    gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
                    return;
                }

                vec2 d = ( vCur.xy / vCur.w - vPrev.xy / vPrev.w ) * 0.5 * uSimSize * uStepRatio;
                d.y = -d.y;                                           // image coordinates: +y down

                vec2 uv = d / uScale;
                float rad = length( uv );
                float fk = ( atan( -uv.y, -uv.x ) / 3.14159265358979 + 1.0 ) * 0.5 * float( NCOLS - 1 );
                int k0 = int( floor( fk ) );
                int k1 = k0 + 1 >= NCOLS ? 0 : k0 + 1;
                vec3 col = mix( uWheel[ k0 ], uWheel[ k1 ], fk - floor( fk ) );

                col = rad <= 1.0 ? 1.0 - rad * ( 1.0 - col ) : col * 0.75;
                gl_FragColor = vec4( col, 1.0 );
            }`
    });
}

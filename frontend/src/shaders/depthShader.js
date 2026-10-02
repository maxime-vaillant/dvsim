const DepthShader = {

	name: 'DepthShader',

	uniforms: {

		tDepth: { value: null },
        // near / far planes of the camera that rendered the depth texture
        uNear: { value: 0.01 },
        uFar: { value: 10 },
        // distance (scene units) that maps to black
        uRange: { value: 10 },

	},

	vertexShader: /* glsl */`

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader: /* glsl */`

		uniform sampler2D tDepth;
        uniform float uNear;
        uniform float uFar;
        uniform float uRange;

        varying vec2 vUv;

		void main() {

			// The depth buffer is non-linear in [0, 1]: recover the distance along the optical axis.
			float depth = texture2D(tDepth, vUv).r;
			float ndcZ = depth * 2.0 - 1.0;
			float zDepth = (2.0 * uNear * uFar) / (uFar + uNear - ndcZ * (uFar - uNear));

			// near = bright, far = dark, nothing rendered (cleared depth) = black
			float shade = depth >= 1.0 ? 0.0 : 1.0 - clamp(zDepth / uRange, 0.0, 1.0);

			gl_FragColor = vec4(vec3(shade), 1.0);

		}`

};

export default DepthShader;

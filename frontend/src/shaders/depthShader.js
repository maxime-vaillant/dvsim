const DepthShader = {

	name: 'DepthShader',

	uniforms: {

		tDepth: { value: null },
        uNear: { value: 0.1 },
        uFar: { value: 5 },

	},

	vertexShader: /* glsl */`

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader: /* glsl */`
	
		uniform sampler2D tDepth;
        varying vec2 vUv;

		void main() {
		
			float depth = texture2D(tDepth, vUv).r;
			
			gl_FragColor = vec4(vec3(1.0 - depth), 1.0);
			
		}`

};

export default DepthShader;

const EventVisionShader = {

	name: 'EventVisionShader',

	uniforms: {

		'tDiffuse': { value: null },
		'uPrevFrame': { value: null },
        // contrast thresholds on the natural-log intensity change (same meaning as c_pos / c_neg in the simulator)
        'thresholdOn': { value: 0.2 },
        'thresholdOff': { value: 0.2 }

	},

	vertexShader: /* glsl */`

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader: /* glsl */`

		uniform float thresholdOn;
		uniform float thresholdOff;

		uniform sampler2D tDiffuse;
		uniform sampler2D uPrevFrame;

		varying vec2 vUv;

		// log(0) is -inf: the epsilon keeps black pixels finite (same role as log_eps in the simulator)
		const float LOG_EPS = 1e-3;

		float getLogLuminance(vec4 color) {
            return log(0.299 * color.r + 0.587 * color.g + 0.114 * color.b + LOG_EPS);
        }

		void main() {

            vec4 prevColor = texture2D(uPrevFrame, vUv);
			vec4 curColor = texture2D(tDiffuse, vUv);

			float logDiff = getLogLuminance(curColor) - getLogLuminance(prevColor);

			vec3 color;
			if (logDiff > thresholdOn) {
			    color = vec3(0, 0, 1);
			} else if (logDiff < -thresholdOff) {
			    color = vec3(1, 0, 0);
			} else {
			    color = vec3(1, 1, 1);
			}

			gl_FragColor = vec4(color, 1.0);
		}`

};

export default EventVisionShader;

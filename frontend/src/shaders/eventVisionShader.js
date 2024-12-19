const EventVisionShader = {

	name: 'EventVisionShader',

	uniforms: {

		'tDiffuse': { value: null },
		'uPrevFrame': { value: null },
        'threshold': { value: 0.1 }

	},

	vertexShader: /* glsl */`

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader: /* glsl */`

		uniform float threshold;

		uniform sampler2D tDiffuse;
		uniform sampler2D uPrevFrame;

		varying vec2 vUv;
		
		float getLuminance(vec4 color) {
            return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
        }

		void main() {
            
            vec4 prevColor = texture2D(uPrevFrame, vUv);
			vec4 curColor = texture2D(tDiffuse, vUv);
			
			float curLuminance = getLuminance(curColor);
            float prevLuminance = getLuminance(prevColor);
			
			float luminanceDiff = log(curLuminance) - log(prevLuminance);
			
			vec3 color;
			if (luminanceDiff > threshold) {
			    color = vec3(0, 0, 1);
			} else if (luminanceDiff < -threshold) {
			    color = vec3(1, 0, 0); 
			} else {
			    color = vec3(1, 1, 1);
			}
			
			gl_FragColor = vec4(color, 1.0); 
		}`

};

export default EventVisionShader;
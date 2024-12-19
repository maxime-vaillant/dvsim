import { THREE_VARIABLES } from "./variables";

function overviewRender() {
    THREE_VARIABLES.renderer.setViewport(0, window.innerHeight / 2, window.innerWidth / 2, window.innerHeight / 2);
    THREE_VARIABLES.renderer.setScissor(0, window.innerHeight / 2, window.innerWidth / 2, window.innerHeight / 2);
    THREE_VARIABLES.renderer.setScissorTest(true);
    THREE_VARIABLES.renderer.render( THREE_VARIABLES.scene, THREE_VARIABLES.overviewCamera );
}

function viewRender() {
    THREE_VARIABLES.renderer.setViewport(
        window.innerWidth / 2, window.innerHeight / 2, window.innerWidth / 2, window.innerHeight / 2
    );
    THREE_VARIABLES.renderer.setScissor(
        window.innerWidth / 2, window.innerHeight / 2, window.innerWidth / 2, window.innerHeight / 2
    );
    THREE_VARIABLES.renderer.setScissorTest(true);
    THREE_VARIABLES.renderer.render( THREE_VARIABLES.scene, THREE_VARIABLES.viewCamera );

    THREE_VARIABLES.renderer.setViewport( window.innerWidth / 2, 0, window.innerWidth / 2, window.innerHeight / 2 );
    THREE_VARIABLES.renderer.setScissor( window.innerWidth / 2, 0, window.innerWidth / 2, window.innerHeight / 2 );
    THREE_VARIABLES.renderer.setScissorTest(true);
    THREE_VARIABLES.eventVisionComposer.render();

    THREE_VARIABLES.renderer.setViewport( 0, 0, window.innerWidth / 2, window.innerHeight / 2 );
    THREE_VARIABLES.renderer.setScissor( 0, 0, window.innerWidth / 2, window.innerHeight / 2 );
    THREE_VARIABLES.renderer.setScissorTest(true);
    THREE_VARIABLES.depthComposer.render();

    THREE_VARIABLES.renderer.setRenderTarget( THREE_VARIABLES.renderTarget );
    THREE_VARIABLES.renderer.render( THREE_VARIABLES.scene, THREE_VARIABLES.viewCamera );

    THREE_VARIABLES.eventVisionPass.uniforms.uPrevFrame.value = THREE_VARIABLES.renderTarget.texture;
    THREE_VARIABLES.renderer.setRenderTarget(null);
}

export function render() {
    overviewRender();

    if ( THREE_VARIABLES.viewCamera ) {
        viewRender();
    }
}
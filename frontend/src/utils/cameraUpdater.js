import { THREE_VARIABLES } from "../components/variables.js";

export function updateViewCamera( position, lookAt ) {
    THREE_VARIABLES.viewCamera.position.copy( position );
    THREE_VARIABLES.viewCamera.lookAt( lookAt );
}
import { THREE_VARIABLES } from "./variables";

const ROOM_DIAGONAL_M = 20;   // up to this size the overview keeps its room-scale framing

// The overview is framed for rooms (5 m above the origin, 100 m far plane, zoom up to 20 m); a larger scene
// (a street, a campus) gets a camera that sees all of it, and the camera marker grows with it.
export function frameOverview( center, diagonal ) {
    const camera = THREE_VARIABLES.overviewCamera;
    const controls = THREE_VARIABLES.controls;
    const large = diagonal > ROOM_DIAGONAL_M;

    if ( large ) {
        camera.position.set( center.x, center.y + diagonal * 0.6, center.z + diagonal * 0.6 );
        controls.target.copy( center );
    } else {
        camera.position.set( 0, 5, 0 );
        controls.target.set( 0, 0, 0 );
    }

    camera.far = large ? diagonal * 4 : 100;
    camera.updateProjectionMatrix();
    controls.maxDistance = large ? diagonal * 3 : 20;
    controls.update();

    THREE_VARIABLES.markerScale = large ? diagonal / ROOM_DIAGONAL_M : 1;
}

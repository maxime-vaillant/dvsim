"""Scene loading (trimesh based). See docs/architecture.md."""
from backend.scene.loader import load_scene, object_matrix
from backend.scene.scene import DEFAULT_BASE_COLOR, Scene, SceneObject

__all__ = ["Scene", "SceneObject", "load_scene", "object_matrix", "DEFAULT_BASE_COLOR"]

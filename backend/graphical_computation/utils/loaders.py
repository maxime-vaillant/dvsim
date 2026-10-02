from pathlib import Path

SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"


def load_shader_source(name: str) -> str:
    """Read a shader from the package's ``shaders/`` directory (independent of the CWD)."""
    return (SHADER_DIR / name).read_text()

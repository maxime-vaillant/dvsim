"""Scene upload validation and storage.

Client filenames never reach the disk: files are stored as ``scene.<ext>`` / ``texture.<ext>`` in a directory of
their own. ``load_scene`` (trimesh) follows file references found inside model files, so uploads are scanned and
anything that could point outside that directory is refused:

* ``.gltf`` is refused (external ``uri``); ``.glb`` is accepted when every ``uri`` in its JSON chunk is a ``data:`` URI;
* ``.obj`` is refused if it contains ``mtllib`` (anywhere, trimesh does a substring search) or ``map_*`` directives;
* ``.ply`` is refused if its header names a ``TextureFile``;
* textures must be real PNG / JPEG images of a sane size (the texture is passed to the loader explicitly).

A *bundle* (an ``.obj`` with its ``.mtl`` and several textures, possibly several ``.obj`` files) is the one case where
model files may reference other files. Client names still never reach the disk: every file is stored under a generated
name (``model_0.obj``, ``material_0.mtl``, ``texture_0.png``...), and the references are rewritten to those names while
the files are copied: the ``mtllib`` line of each OBJ and the texture directives (``map_Kd``...) of each MTL. A
reference may carry directories, backslashes, spaces or texture options (``-s 1 1 1``, dropped); it is matched to an
uploaded file by its file name. A diffuse map (``map_Kd``) that was not uploaded is an error that names the file;
other maps are dropped (the renderer only uses diffuse colour).
"""
from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple

from fastapi import HTTPException

MODEL_EXTS = (".obj", ".glb", ".ply")
TEXTURE_EXTS = (".png", ".jpg", ".jpeg")
MAX_TEXTURE_PIXELS = 100_000_000
MAX_BUNDLE_FILES = 64
MAX_MTL_BYTES = 16 * 1024 * 1024
_MAX_LINE = 1 << 20           # an OBJ line longer than this is copied in pieces (no line-based rewriting inside)
_CHUNK = 1 << 20
_TAIL = 128  # bytes carried over between chunks so a forbidden token split across two reads is still found
_GLB_JSON_MAX = 64 * 1024 * 1024
_FORBIDDEN = {
    ".obj": (
        re.compile(rb"mtllib|^[ \t]*map_\w+", re.I | re.M),
        "OBJ files must not reference material libraries (mtllib / map_*): upload the OBJ and its texture separately",
    ),
    ".ply": (re.compile(rb"texturefile", re.I), "PLY files must not reference a TextureFile"),
}


def _reject(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def _copy(src: BinaryIO, dest: Path, limit: int, ext: Optional[str] = None) -> int:
    """Stream ``src`` to ``dest`` in chunks, enforcing the size limit and the forbidden-content scan."""
    forbidden = _FORBIDDEN.get(ext) if ext else None
    size, tail = 0, b""
    with open(dest, "wb") as out:
        while True:
            chunk = src.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise _reject(413, f"file too large (limit {limit // (1024 * 1024)} MB)")
            if forbidden is not None:
                window = tail + chunk
                if forbidden[0].search(window):
                    raise _reject(422, forbidden[1])
                tail = window[-_TAIL:]
            out.write(chunk)
    if size == 0:
        raise _reject(422, "empty file")
    return size


def _external_uri(document: Any) -> Optional[str]:
    """First ``uri`` string of a glTF JSON document that is not a ``data:`` URI."""
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "uri" and isinstance(value, str) and not value.startswith("data:"):
                    return value
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return None


def _check_glb(path: Path) -> None:
    try:
        with open(path, "rb") as fh:
            head = fh.read(20)
            if len(head) < 20 or head[:4] != b"glTF":
                raise ValueError("not a GLB file")
            version, _, length, kind = struct.unpack("<IIII", head[4:20])
            if version != 2 or kind != 0x4E4F534A:
                raise ValueError("expected glTF 2.0 with a JSON first chunk")
            if length > _GLB_JSON_MAX or 20 + length > path.stat().st_size:
                raise ValueError("bad JSON chunk length")
            document = json.loads(fh.read(length))
    except (ValueError, struct.error, OSError) as exc:
        raise _reject(422, f"invalid .glb: {exc}") from None
    uri = _external_uri(document)
    if uri is not None:
        raise _reject(422, "GLB files must be self-contained (embedded buffers and images), found external uri")


def _check_image(path: Path) -> str:
    from PIL import Image

    try:
        with Image.open(path) as img:
            fmt, (width, height) = img.format, img.size
    except Exception:
        raise _reject(422, "texture is not a valid PNG or JPEG image") from None
    if fmt not in ("PNG", "JPEG"):
        raise _reject(422, "texture must be a PNG or JPEG image")
    if width * height > MAX_TEXTURE_PIXELS:
        raise _reject(422, f"texture too large ({width}x{height} px)")
    return ".png" if fmt == "PNG" else ".jpg"


def store_scene_files(
    dest: Path,
    model: BinaryIO,
    model_name: Optional[str],
    texture: Optional[BinaryIO],
    texture_name: Optional[str],
    max_bytes: int,
) -> Dict[str, Any]:
    """Validate and write an upload into ``dest``; returns ``{"model", "texture", "size_bytes"}`` (stored names)."""
    ext = Path(model_name or "").suffix.lower()
    if ext == ".gltf":
        raise _reject(422, ".gltf files are not accepted (they reference external files): upload a self-contained .glb")
    if ext not in MODEL_EXTS:
        raise _reject(422, f"unsupported scene file type {ext or '(none)'}; allowed: {', '.join(MODEL_EXTS)}")
    if texture is not None and ext != ".obj":
        raise _reject(422, "a separate texture is only supported with .obj scenes")

    stored = {"model": f"scene{ext}", "texture": None}
    size = _copy(model, dest / stored["model"], max_bytes, ext)
    if ext == ".glb":
        _check_glb(dest / stored["model"])
    if texture is not None:
        texture_ext = Path(texture_name or "").suffix.lower()
        if texture_ext and texture_ext not in TEXTURE_EXTS:
            raise _reject(422, f"unsupported texture file type {texture_ext}; allowed: {', '.join(TEXTURE_EXTS)}")
        temp = dest / "texture.tmp"
        size += _copy(texture, temp, max_bytes)
        stored["texture"] = f"texture{_check_image(temp)}"
        os.replace(temp, dest / stored["texture"])
    stored["size_bytes"] = size
    return stored


# ---- bundles: OBJ + MTL + textures (+ more OBJ) ----------------------------------------------------------------

# options of a texture directive and how many values follow them (-o / -s / -t take up to three numbers)
_MTL_OPTIONS = {"-blendu": 1, "-blendv": 1, "-cc": 1, "-clamp": 1, "-imfchan": 1, "-mm": 2, "-o": 3, "-s": 3,
                "-t": 3, "-texres": 1, "-bm": 1, "-type": 1}
_TEXTURE_DIRECTIVE = re.compile(r"^(\s*)(map_\w+|bump|disp|decal|refl)(\s+)(.*?)\s*$", re.I)


def _file_name(reference: str) -> str:
    """The file name of a reference that may carry directories (/ or \\)."""
    return PurePosixPath(reference.replace("\\", "/")).name


_OPTION_VALUE = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?|on|off|[a-z]", re.I)


def _texture_reference(rest: str) -> Tuple[int, str]:
    """(offset, file name) of a texture directive's argument: the options (``-s 1 1 1``) are skipped, the rest of
    the line is the file name (it may contain spaces)."""
    tokens = list(re.finditer(r"\S+", rest))
    i = 0
    while i < len(tokens) and tokens[i].group().lower() in _MTL_OPTIONS:
        budget = _MTL_OPTIONS[tokens[i].group().lower()]
        i += 1
        while budget and i < len(tokens) - 1 and _OPTION_VALUE.fullmatch(tokens[i].group()):
            i += 1                                   # (the last token is the file name, never an option value)
            budget -= 1
    if i >= len(tokens):
        return 0, ""
    start = tokens[i].start()
    return start, rest[start:]


def _lookup(table: Dict[str, str], reference: str) -> Optional[str]:
    """Stored name of the uploaded file a reference points to (by file name, case-insensitively)."""
    return table.get(_file_name(reference).lower())


def _rewrite_mtl(text: str, textures: Dict[str, str], name: str) -> str:
    """The MTL with every texture directive pointing at a stored texture; unknown non-diffuse maps are dropped."""
    out: List[str] = []
    for line in text.splitlines():
        match = _TEXTURE_DIRECTIVE.match(line)
        if match is None:
            out.append(line)
            continue
        indent, directive, gap, rest = match.groups()
        _, reference = _texture_reference(rest)
        stored = _lookup(textures, reference) if reference else None
        if stored is not None:                       # the options (scale, offset...) are dropped: trimesh would take
            out.append(f"{indent}{directive}{gap}{stored}")      # them for the file name, and the renderer ignores them
        elif directive.lower() == "map_kd":
            raise _reject(422, f"{name} uses the texture '{_file_name(reference)}': upload it together with the model")
        # any other map whose image was not uploaded is dropped
    return "\n".join(out) + "\n"


def _copy_obj(src: BinaryIO, dest: Path, limit: int, materials: Dict[str, str], name: str) -> int:
    """Copy an OBJ, rewriting its ``mtllib`` line(s) to stored material library names. Returns the size copied."""
    size, at_start = 0, True
    with open(dest, "wb") as out:
        while True:
            line = src.readline(_MAX_LINE)
            if not line:
                break
            size += len(line)
            if size > limit:
                raise _reject(413, f"file too large (limit {limit // (1024 * 1024)} MB)")
            if at_start and b"mtllib" in line.lower():
                text = line.decode("utf-8", "replace").strip()
                if text.lower().startswith("mtllib"):
                    references = text[6:].strip()
                    whole = _lookup(materials, references)          # a file name with spaces, or several names
                    stored = [whole] if whole else [_lookup(materials, token) for token in references.split()]
                    if not references or any(item is None for item in stored):
                        raise _reject(422, f"{name} refers to the material file '{_file_name(references)}': "
                                           "upload it together with the model")
                    line = f"mtllib {' '.join(stored)}\n".encode()
                else:
                    line = b"#\n"          # trimesh would find a "mtllib" inside a comment too: drop the line
            out.write(line)
            at_start = line.endswith(b"\n")
    if size == 0:
        raise _reject(422, "empty file")
    return size


def _unique_names(files: Sequence[Tuple[str, Any]]) -> None:
    seen = set()
    for name, _ in files:
        key = _file_name(name).lower()
        if not key or key in seen:
            raise _reject(422, f"two uploaded files are called '{_file_name(name)}' (names must be distinct)")
        seen.add(key)


def store_scene_bundle(dest: Path, files: Sequence[Tuple[str, BinaryIO]], max_bytes: int) -> Dict[str, Any]:
    """Validate and write a multi-file upload (OBJ files, MTL files, PNG / JPEG textures) into ``dest``.

    Returns ``{"model", "models", "texture": None, "files", "size_bytes"}`` with the stored names."""
    if len(files) > MAX_BUNDLE_FILES:
        raise _reject(422, f"too many files ({len(files)}; the limit is {MAX_BUNDLE_FILES})")
    _unique_names(files)
    kinds = {".obj": [], ".mtl": [], "image": []}
    for name, handle in files:
        ext = Path(_file_name(name)).suffix.lower()
        kind = ext if ext in (".obj", ".mtl") else "image" if ext in TEXTURE_EXTS else None
        if kind is None:
            raise _reject(422, f"unsupported file '{_file_name(name)}' in a scene bundle; allowed: "
                               f".obj, .mtl, {', '.join(TEXTURE_EXTS)}")
        kinds[kind].append((name, handle))
    if not kinds[".obj"]:
        raise _reject(422, "a scene bundle needs at least one .obj file")

    size, stored_files = 0, []
    textures: Dict[str, str] = {}
    for index, (name, handle) in enumerate(kinds["image"]):          # textures first: the MTL files point at them
        temp = dest / f"texture_{index}.tmp"
        size += _copy(handle, temp, max_bytes)
        final = f"texture_{index}{_check_image(temp)}"
        os.replace(temp, dest / final)
        textures[_file_name(name).lower()] = final
        stored_files.append(final)
    materials: Dict[str, str] = {}
    for index, (name, handle) in enumerate(kinds[".mtl"]):
        final = f"material_{index}.mtl"
        raw = handle.read(MAX_MTL_BYTES + 1)
        if len(raw) > MAX_MTL_BYTES:
            raise _reject(413, f"material file too large (limit {MAX_MTL_BYTES // (1024 * 1024)} MB)")
        if not raw:
            raise _reject(422, "empty file")
        size += len(raw)
        (dest / final).write_text(_rewrite_mtl(raw.decode("utf-8", "replace"), textures, _file_name(name)),
                                  encoding="utf-8")
        materials[_file_name(name).lower()] = final
        stored_files.append(final)
    models = []
    for index, (name, handle) in enumerate(kinds[".obj"]):
        final = f"model_{index}.obj"
        size += _copy_obj(handle, dest / final, max_bytes, materials, _file_name(name))
        models.append(final)
        stored_files.append(final)
    if size > max_bytes:
        raise _reject(413, f"the bundle is too large (limit {max_bytes // (1024 * 1024)} MB in total)")
    return {"model": models[0], "models": models, "texture": None, "files": stored_files, "size_bytes": size}

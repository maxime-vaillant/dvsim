// A scene made of several files: OBJ files, their MTL files and the textures. DOM free, unit-tested in node.
// The server does the same matching (backend/api/uploads.py): a reference is resolved by its file name, whatever
// directories, backslashes, spaces or texture options it carries.

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg'];
// options of a texture directive and how many values follow them
const MTL_OPTIONS = { '-blendu': 1, '-blendv': 1, '-cc': 1, '-clamp': 1, '-imfchan': 1, '-mm': 2, '-o': 3, '-s': 3, '-t': 3, '-texres': 1, '-bm': 1, '-type': 1 };
const OPTION_VALUE = /^([-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?|on|off|[a-z])$/i;
const TEXTURE_DIRECTIVE = /^\s*(map_\w+|bump|disp|decal|refl)\s+(.*?)\s*$/i;

/** The file name of a reference that may carry directories (/ or \). */
export function fileName(reference) {
    return String(reference).replace(/\\/g, '/').split('/').pop();
}

const extensionOf = name => {
    const dot = name.lastIndexOf('.');

    return dot < 0 ? '' : name.slice(dot).toLowerCase();
};

/** { objs, mtls, images, others } of picked files ({ name } objects), each list in the given order. */
export function classifyFiles(files) {
    const result = { objs: [], mtls: [], images: [], others: [] };

    files.forEach(file => {
        const extension = extensionOf(file.name);

        if (extension === '.obj') result.objs.push(file);
        else if (extension === '.mtl') result.mtls.push(file);
        else if (IMAGE_EXTENSIONS.includes(extension)) result.images.push(file);
        else result.others.push(file);
    });

    return result;
}

/** File names after `mtllib` in an OBJ text (a whole-line name, so spaces are kept). */
export function referencedMaterialLibraries(objText) {
    const names = [];

    objText.split(/\r?\n/).forEach(line => {
        const match = /^\s*mtllib\s+(.+?)\s*$/i.exec(line);

        if (match) names.push(fileName(match[1]));
    });

    return names;
}

/** The argument of a texture directive without its options: the texture's file name. */
function textureArgument(rest) {
    const tokens = [...rest.matchAll(/\S+/g)];
    let i = 0;

    while (i < tokens.length && tokens[i][0].toLowerCase() in MTL_OPTIONS) {
        let budget = MTL_OPTIONS[tokens[i][0].toLowerCase()];

        i += 1;
        while (budget && i < tokens.length - 1 && OPTION_VALUE.test(tokens[i][0])) {
            i += 1;
            budget -= 1;
        }
    }

    return i < tokens.length ? rest.slice(tokens[i].index) : '';
}

/** { diffuse, other }: texture file names used by an MTL text (map_Kd versus the other maps). */
export function referencedTextures(mtlText) {
    const result = { diffuse: [], other: [] };

    mtlText.split(/\r?\n/).forEach(line => {
        const match = TEXTURE_DIRECTIVE.exec(line);

        if (!match) return;

        const name = textureArgument(match[2]);

        if (name) (match[1].toLowerCase() === 'map_kd' ? result.diffuse : result.other).push(fileName(name));
    });

    return result;
}

/**
 * Files a scene needs but the user did not pick: [{ name, neededBy }]. `objs` / `mtls` are { name, text } (an OBJ's
 * text may be just its beginning: `mtllib` comes first); `available` are the names of every picked file.
 */
export function missingFiles({ objs, mtls, available }) {
    const have = new Set(available.map(name => fileName(name).toLowerCase()));
    const mtlByName = new Map(mtls.map(mtl => [fileName(mtl.name).toLowerCase(), mtl]));
    const missing = [];
    const add = (name, neededBy) => {
        if (!missing.some(item => item.name.toLowerCase() === name.toLowerCase())) missing.push({ name, neededBy });
    };

    objs.forEach(obj => referencedMaterialLibraries(obj.text).forEach(library => {
        if (!mtlByName.has(library.toLowerCase())) add(library, obj.name);
    }));
    mtls.forEach(mtl => referencedTextures(mtl.text).diffuse.forEach(texture => {
        if (!have.has(texture.toLowerCase())) add(texture, mtl.name);
    }));

    return missing;
}

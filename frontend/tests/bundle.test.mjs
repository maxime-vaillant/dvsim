// Multi-file scenes: which files were picked, what the OBJ / MTL refer to, what is missing.
import test from 'node:test';
import assert from 'node:assert/strict';

import { classifyFiles, fileName, missingFiles, referencedMaterialLibraries, referencedTextures } from '../src/utils/sceneBundle.js';

test('file names ignore directories, backslashes and drives', () => {
    assert.equal(fileName('Textures/colormap.png'), 'colormap.png');
    assert.equal(fileName('C:\\scans\\my tex.PNG'), 'my tex.PNG');
    assert.equal(fileName('plain.jpg'), 'plain.jpg');
});

test('picked files are sorted by kind, in the given order', () => {
    const names = ['b.obj', 'A.MTL', 't1.jpg', 'a.obj', 't2.PNG', 'notes.txt', 'archive.zip', 'x.jpeg'];
    const { objs, mtls, images, others } = classifyFiles(names.map(name => ({ name })));

    assert.deepEqual(objs.map(f => f.name), ['b.obj', 'a.obj']);
    assert.deepEqual(mtls.map(f => f.name), ['A.MTL']);
    assert.deepEqual(images.map(f => f.name), ['t1.jpg', 't2.PNG', 'x.jpeg']);
    assert.deepEqual(others.map(f => f.name), ['notes.txt', 'archive.zip']);
});

test('mtllib: the whole name (spaces kept), CRLF files, none', () => {
    assert.deepEqual(referencedMaterialLibraries('# scan\r\nmtllib my materials.mtl\r\nv 0 0 0\r\n'), ['my materials.mtl']);
    assert.deepEqual(referencedMaterialLibraries('mtllib dir\\a.mtl\nmtllib b.mtl\n'), ['a.mtl', 'b.mtl']);
    assert.deepEqual(referencedMaterialLibraries('v 0 0 0\nf 1 1 1\n'), []);
});

test('MTL textures: options are skipped, directories dropped, diffuse told apart', () => {
    const mtl = [
        'newmtl a', 'map_Kd Textures\\red tex.png',
        'newmtl b', 'map_Kd -s 1 1 1 -clamp on C:/x/blue.jpg', 'map_Ks spec.png', 'bump -bm 0.5 bump.png',
        'newmtl c', 'Kd 1 1 1', 'map_Kd -o 0 0 plain.png'
    ].join('\r\n');
    const { diffuse, other } = referencedTextures(mtl);

    assert.deepEqual(diffuse, ['red tex.png', 'blue.jpg', 'plain.png']);
    assert.deepEqual(other, ['spec.png', 'bump.png']);
    assert.deepEqual(referencedTextures('newmtl x\nKd 1 1 1\n'), { diffuse: [], other: [] });
});

test('missing files: the material library of each OBJ and the diffuse textures of each MTL', () => {
    const objs = [{ name: 'a.obj', text: 'mtllib scene.mtl\n' }, { name: 'b.obj', text: 'mtllib other.mtl\n' }];
    const mtls = [{ name: 'scene.mtl', text: 'map_Kd t1.jpg\nmap_Kd t2.jpg\nmap_Ks s.png\n' }];
    const all = missingFiles({ objs, mtls, available: ['a.obj', 'b.obj', 'scene.mtl', 'T1.JPG'] });

    assert.deepEqual(all, [{ name: 'other.mtl', neededBy: 'b.obj' }, { name: 't2.jpg', neededBy: 'scene.mtl' }]);
    assert.deepEqual(missingFiles({ objs: [objs[0]], mtls, available: ['scene.mtl', 't1.jpg', 't2.jpg'] }), []);
    assert.deepEqual(missingFiles({ objs: [{ name: 'c.obj', text: 'v 0 0 0\n' }], mtls: [], available: [] }), [], 'no mtllib');
});

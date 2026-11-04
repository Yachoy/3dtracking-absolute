# Полная треугольная сетка CargoTruck.fbx в системе оболочек truck_hulls.json (см, оси UE: вперёд, вправо,
# вверх, начало — передний нижний край кабины) для --appearance-target (PicoPose) без Assimp.
# python tools/truck_mesh.py CargoTruck.fbx models/truck  ->  models/truck/{truck.ply,target.json}
import json
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
import trimesh

data = Path(sys.argv[1]).read_bytes()
out_dir = Path(sys.argv[2])
assert data[:20] == b'Kaydara FBX Binary  ', data[:30]
big = struct.unpack('<I', data[23:27])[0] >= 7500


def rd_node(p):
    if big: end, nprop, plen = struct.unpack('<QQQ', data[p:p+24]); p += 24
    else: end, nprop, plen = struct.unpack('<III', data[p:p+12]); p += 12
    if end == 0: return None, p
    nl = data[p]; name = data[p+1:p+1+nl].decode(); p += 1 + nl
    props = []
    for _ in range(nprop):
        t = chr(data[p]); p += 1
        if t in 'YCILFD':
            fmt = {'Y': '<h', 'C': '<?', 'I': '<i', 'L': '<q', 'F': '<f', 'D': '<d'}[t]; sz = struct.calcsize(fmt)
            props.append(struct.unpack(fmt, data[p:p+sz])[0]); p += sz
        elif t in 'fdlib':
            n, enc, cl = struct.unpack('<III', data[p:p+12]); p += 12
            raw = data[p:p+cl]; p += cl
            if enc: raw = zlib.decompress(raw)
            props.append(np.frombuffer(raw, {'f': '<f4', 'd': '<f8', 'l': '<i8', 'i': '<i4', 'b': '?'}[t]))
        elif t in 'SR':
            n = struct.unpack('<I', data[p:p+4])[0]; p += 4; props.append(data[p:p+n]); p += n
    kids = []
    while p < end:
        k, p = rd_node(p)
        if k is None: break
        kids.append(k)
    return (name, props, kids), end


def walk(n, path=''):
    yield path + '/' + n[0], n
    for k in n[2]: yield from walk(k, path + '/' + n[0])


def rotm(r):
    x, y, z = np.radians(r)
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


p = 27; top = []
while p < len(data) - 160:
    n, p2 = rd_node(p)
    if n is None: break
    top.append(n); p = p2

# тот же порядок сопоставления геометрий и моделей, что в experiments/sam2_model/fbxlib.py
models, geoms = [], []
for t in top:
    for path, n in walk(t):
        if path.endswith('Objects/Geometry'):
            kid = {k[0]: k[1][0] for k in n[2] if k[0] in ('Vertices', 'PolygonVertexIndex')}
            if 'Vertices' in kid: geoms.append((kid['Vertices'].reshape(-1, 3), kid.get('PolygonVertexIndex')))
        if path.endswith('Objects/Model'):
            d = {'name': n[1][1].split(b'\x00')[0].decode(), 'T': np.zeros(3), 'R': np.zeros(3), 'S': np.ones(3)}
            for k in n[2]:
                if k[0] == 'Properties70':
                    for pp in k[2]:
                        key = pp[1][0]
                        if key == b'Lcl Translation': d['T'] = np.array(pp[1][4:7])
                        if key == b'Lcl Rotation': d['R'] = np.array(pp[1][4:7])
                        if key == b'Lcl Scaling': d['S'] = np.array(pp[1][4:7])
            models.append(d)

hulls = json.loads((Path(__file__).resolve().parents[1] / 'experiments/sam2_model/truck_hulls.json').read_text())
front = np.array(hulls['front_cm'])
meshes = []
for d, (v, index) in zip(models, geoms):
    w = (rotm(d['R']) @ (v * d['S']).T).T + d['T']
    ue = np.stack([w[:, 2], w[:, 0], w[:, 1]], 1) - front
    faces, poly = [], []
    for i in index:
        poly.append(i if i >= 0 else ~i)
        if i < 0:
            faces += [(poly[0], poly[j], poly[j + 1]) for j in range(1, len(poly) - 1)]
            poly = []
    # перестановка осей циклическая (поворот), поэтому обход граней и нормали наружу сохраняются
    meshes.append(trimesh.Trimesh(ue, np.array(faces), process=False))
    print(d['name'], len(ue), len(faces), ue.min(0).round(0), ue.max(0).round(0))

truck = trimesh.util.concatenate(meshes)
all_hull = np.vstack([np.array(h) for h in hulls['hulls_cm'].values()])
print('bbox mesh', truck.bounds.round(1).tolist(), 'hulls', [all_hull.min(0).tolist(), all_hull.max(0).tolist()])
out_dir.mkdir(parents=True, exist_ok=True)
truck.export(out_dir / 'truck.ply')
# масштаб ×3.1 и опорная точка совпадают с way1_target() в online/target.py
(out_dir / 'target.json').write_text(json.dumps({
    'name': 'CargoTruck', 'mesh': 'truck.ply', 'meters_per_unit': 0.031, 'origin': [0, 0, 0],
    'labels': [2, 5, 7], 'level_surface': True}, indent=2), encoding='utf-8')

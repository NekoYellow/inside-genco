"""Input-only identities and reproducible operating-condition selections."""
import hashlib
import json
from pathlib import Path

import numpy as np

BUS_COLUMNS = ['bus', 'PQ', 'PV', 'REF', 'GS', 'BS', 'vn_kv', 'Pd', 'Qd']
GEN_COLUMNS = ['idx', 'bus', 'in_service', 'is_slack_gen']
Y_COLUMNS = ['Yff_r', 'Yff_i', 'Yft_r', 'Yft_i', 'Ytf_r', 'Ytf_i', 'Ytt_r', 'Ytt_i']
BRANCH_COLUMNS = ['idx', 'from_bus', 'to_bus', 'br_status', 'r', 'x', 'b', 'tap', 'shift'] + Y_COLUMNS
TOPOLOGY_SCHEMA = 'case118-input-identity-v1'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _digest(parts):
    """Fixed columns, sorted device IDs, little-endian float64; no rounding."""
    h = hashlib.sha256(TOPOLOGY_SCHEMA.encode())
    for name, frame in parts:
        values = frame.to_numpy(dtype='<f8', copy=True)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite identity input: {name}')
        values[values == 0] = 0.0  # canonicalize negative zero
        h.update(json.dumps([name, list(frame.columns), list(values.shape)]).encode())
        h.update(values.tobytes(order='C'))
    return h.hexdigest()


def _devices(frame, id_col):
    values = frame[id_col].to_numpy()
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f'Noninteger {id_col}')
    if frame[id_col].duplicated().any():
        raise ValueError(f'Duplicate device {id_col}')
    return frame.sort_values(id_col).reset_index(drop=True)


def network_identity(bus, gen, branch):
    """Never reads solved voltages, generation, or branch-flow targets."""
    b, g, e = _devices(bus, 'bus'), _devices(gen, 'idx'), _devices(branch, 'idx')
    for frame, col in [(g, 'in_service'), (e, 'br_status')]:
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f'Invalid {col}')
    g, e = g[g.in_service == 1], e[e.br_status == 1]
    if not b[['PQ', 'PV', 'REF']].isin([0, 1]).all().all() or not (b[['PQ','PV','REF']].sum(axis=1) == 1).all():
        raise ValueError('Invalid bus type')
    if b.REF.sum() != 1 or g.empty or e.empty:
        raise ValueError('Expected one REF and active devices')
    ids = set(b.bus)
    if not set(g.bus) <= ids or not set(e.from_bus) <= ids or not set(e.to_bus) <= ids:
        raise ValueError('Unknown endpoint')
    node_parts = [('bus', b[['bus','PQ','PV','REF']]), ('gen', g[['idx','bus','is_slack_gen']])]
    topology_parts = node_parts + [('branch', e[['idx','from_bus','to_bus']])]
    topology = _digest(topology_parts)
    electrical = _digest(topology_parts + [('shunt', b[['bus','GS','BS','vn_kv']]),
                                           ('admittance', e[['idx','r','x','b','tap','shift'] + Y_COLUMNS])])
    return {
        'topology_id': topology, 'electrical_id': electrical, 'node_map_id': _digest(node_parts),
        'bus_count': len(b), 'active_gen_count': len(g), 'active_branch_count': len(e),
        'ref_bus_id': int(b.loc[b.REF == 1, 'bus'].iloc[0]),
        'total_pd_MW': float(b.Pd.sum()), 'total_qd_Mvar': float(b.Qd.sum()),
    }


def stable_order(values, seed, namespace):
    return sorted(values, key=lambda x: hashlib.sha256(f'{seed}:{namespace}:{int(x)}'.encode()).hexdigest())


def assign_groups(groups, exposed_groups, seed=20261008, target=400):
    groups, exposed = set(map(int, groups)), set(map(int, exposed_groups))
    if not exposed <= groups:
        raise ValueError('Exposed groups outside official test')
    if len(exposed) > target or len(groups) <= len(exposed):
        raise ValueError('Insufficient unexposed groups for the requested design')
    unexposed = stable_order(groups - exposed, seed, 'group-split')
    n_explore = max(len(exposed), min(target, len(groups) // 2))
    n_confirm = min(target, len(groups) - n_explore)
    add = n_explore - len(exposed)
    exploration = sorted(exposed | set(unexposed[:add]))
    confirmation = sorted(unexposed[add:add+n_confirm])
    reserve = sorted(unexposed[add+n_confirm:])
    return dict(exploration=exploration, confirmation=confirmation, reserve=reserve)


def validate_ids(ids, official_test):
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('Scenario IDs must be a nonempty list of nonnegative integers')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate scenario IDs')
    if not set(ids) <= set(official_test):
        raise ValueError('Scenario selection outside official test')
    return sorted(ids)


def load_selection(path, official_test, root):
    """Verify content/provenance and load rows for an explicit research selection."""
    import pandas as pd
    path, root = Path(path).resolve(), Path(root).resolve()
    spec = json.loads(path.read_text())
    if isinstance(spec, list):
        return validate_ids(spec, official_test), {}, {'path': str(path), 'sha256': file_hash(path)}
    ids = validate_ids(spec['scenario_ids'], official_test)
    for rel, expected in spec['dependencies'].items():
        dep = (root / rel).resolve()
        if not dep.is_relative_to(root) or file_hash(dep) != expected:
            raise ValueError(f'Selection dependency hash mismatch: {rel}')
    index = pd.read_csv(root / spec['index_path']).set_index('scenario_id')
    if not index.index.is_unique or not set(ids) <= set(index.index):
        raise ValueError('Research index does not cover unique selected IDs')
    rows = index.loc[ids]
    allowed_splits = spec.get('research_splits')
    if allowed_splits is None:
        if set(rows.research_split) != {spec['research_split']}:
            raise ValueError('Selection crosses research splits')
    elif set(rows.research_split) != set(allowed_splits):
        raise ValueError('Selection does not cover the declared research splits')
    if set(rows.topology_id) != {spec['topology_id']}:
        raise ValueError('Selection crosses target topologies')
    info = {'path': str(path), 'sha256': file_hash(path), 'name': spec['name'],
            'research_split': spec['research_split'], 'dependencies': spec['dependencies']}
    if allowed_splits is not None:
        info['research_splits'] = allowed_splits
    return ids, rows.to_dict('index'), info

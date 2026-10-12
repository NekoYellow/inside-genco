#!/usr/bin/env python3
"""Evaluate the original GENCO PF forward on the published case118 test split."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/gridfm-graphkit'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

from download_assets import verify
from research_data import load_selection, network_identity


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_splits(path):
    splits = json.loads(Path(path).read_text())
    seen = set()
    for name in ('train', 'val', 'test'):
        ids = splits[name]
        if not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError(f'Invalid {name} scenario IDs')
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Duplicate scenario IDs or train/val/test overlap')
        seen.update(ids)
    return splits


def partition_ranges(entries):
    """Read actual parquet statistics; never assume partition_id == scenario_id//N."""
    import pyarrow.parquet as pq
    by_partition = {}
    for entry in entries:
        if not entry['path'].startswith('bus_data.parquet/'):
            continue
        path = ROOT / entry['local']
        if not path.is_file():
            continue
        pf = pq.ParquetFile(path)
        col = pf.schema_arrow.names.index('scenario')
        stats = [pf.metadata.row_group(i).column(col).statistics for i in range(pf.num_row_groups)]
        if not stats or any(s is None or not s.has_min_max for s in stats):
            values = pf.read(columns=['scenario']).column('scenario').to_numpy()
            lo, hi = int(values.min()), int(values.max())
        else:
            lo, hi = int(min(s.min for s in stats)), int(max(s.max for s in stats))
        partition = path.parent.name
        if partition in by_partition:
            old_lo, old_hi = by_partition[partition]
            by_partition[partition] = min(lo, old_lo), max(hi, old_hi)
        else:
            by_partition[partition] = lo, hi
    ranges = sorted((lo, hi, name) for name, (lo, hi) in by_partition.items())
    if any(a[1] >= b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError('Overlapping scenario ranges in parquet partitions')
    return ranges


def plan_partitions(ids, entries):
    remaining = set(ids)
    plan = []
    for lo, hi, partition in partition_ranges(entries):
        wanted = sorted(i for i in remaining if lo <= i <= hi)
        if wanted:
            plan.append((partition, wanted))
            remaining.difference_update(wanted)
    if remaining:
        raise FileNotFoundError(f'Missing data for {len(remaining)} selected scenarios. Run scripts/download_assets.py first.')
    by_partition = {}
    for entry in entries:
        parts = entry['path'].split('/')
        if len(parts) == 3 and parts[0] in ('bus_data.parquet', 'gen_data.parquet', 'branch_data.parquet'):
            by_partition.setdefault(parts[1], []).append(entry)
    required = []
    for partition, _ in plan:
        part_entries = by_partition[partition]
        if {e['path'].split('/')[0] for e in part_entries} != {'bus_data.parquet','gen_data.parquet','branch_data.parquet'}:
            raise ValueError(f'Missing table for {partition}')
        required.extend(part_entries)
    return plan, required


def graph_batches(plan, config, normalizer, batch_size, metadata=None, raw_dir=None):
    import torch
    import pandas as pd
    from torch_geometric.data import Batch
    from gridfm_graphkit.datasets.hetero_preprocess import (
        build_hetero_data_for_scenario, merge_agg_gen_into_bus,
    )
    from gridfm_graphkit.io.param_handler import get_task_transforms
    transform = get_task_transforms(config)
    raw = Path(raw_dir) if raw_dir is not None else ROOT / 'assets/data/case118_ieee/raw'
    graphs, groups, topologies = [], [], []
    for partition, ids in plan:
        tables = []
        for table in ('bus', 'gen', 'branch'):
            frame = pd.read_parquet(raw / f'{table}_data.parquet' / partition,
                                    filters=[('scenario', 'in', ids)])
            tables.append(frame)
        bus, gen, branch = tables
        if set(bus.scenario.unique()) != set(ids):
            raise ValueError(f'Scenario coverage mismatch in {partition}')
        bus = merge_agg_gen_into_bus(bus, gen)
        bg, gg, eg = [df.groupby('scenario', sort=False) for df in (bus, gen, branch)]
        for sid in ids:
            b, g, e = bg.get_group(sid), gg.get_group(sid), eg.get_group(sid)
            group = b.load_scenario_idx.unique()
            if len(group) != 1:
                raise ValueError(f'Scenario {sid} has ambiguous load group')
            identity = network_identity(b, g, e)
            topology = identity['topology_id']
            if metadata:
                expected = metadata[sid]
                if any(identity[k] != expected[k] for k in ('topology_id','electrical_id','node_map_id')) or int(group[0]) != expected['load_scenario_idx']:
                    raise ValueError(f'Research identity mismatch: {sid}')
            graph = build_hetero_data_for_scenario(sid, b, g, e)
            normalizer.transform(graph)
            graph = transform(graph)
            # Stable raw IDs follow the official transform's preserved row order.
            graph['bus'].device_id = torch.tensor(b.bus.to_numpy(), dtype=torch.long)
            graph['gen'].device_id = torch.tensor(g.loc[g.in_service == 1, 'idx'].to_numpy(), dtype=torch.long)
            edge_ids = torch.tensor(e.loc[e.br_status == 1, 'idx'].to_numpy(), dtype=torch.long)
            graph['bus','connects','bus'].device_id = torch.cat([edge_ids, edge_ids])
            graphs.append(graph)
            groups.append(int(group[0]))
            topologies.append(topology)
            if len(graphs) == batch_size:
                yield Batch.from_data_list(graphs), groups, topologies
                graphs, groups, topologies = [], [], []
    if graphs:
        yield Batch.from_data_list(graphs), groups, topologies


def residuals(output, batch):
    from gridfm_graphkit.models.utils import ComputeBranchFlow, ComputeNodeInjection, ComputeNodeResiduals
    key = ('bus', 'connects', 'bus')
    edge = batch.edge_index_dict[key]
    p, q = ComputeBranchFlow()(output['bus'], edge, batch.edge_attr_dict[key])
    p, q = ComputeNodeInjection()(p, q, edge, batch['bus'].num_nodes)
    return ComputeNodeResiduals()(p, q, output['bus'], batch.x_dict['bus'])


def to_metric_dtype(batch, dtype):
    return batch.clone().apply(lambda t: t.to(dtype=dtype) if t.is_floating_point() else t)


def physicalize_output(output, batch, normalizer, metric_dtype='float32'):
    """Return cloned inputs and predictions converted to physical units."""

    import torch

    if metric_dtype == 'float64':
        batch = to_metric_dtype(batch, torch.float64)
        output = {k: v.to(dtype=torch.float64) for k, v in output.items()}
    physical = batch.clone()
    physical_output = {k: v.clone() for k, v in output.items()}
    normalizer.inverse_transform(physical)
    normalizer.inverse_output(physical_output, physical)
    return physical_output, physical


def scenario_metrics(output, batch, normalizer, groups, topologies, metric_dtype='float32'):
    import torch
    from gridfm_graphkit.datasets.globals import VM_H, VA_H, VM_OUT, VA_OUT
    # Compute in physical units exactly as PowerFlowTask.test_step does. Clone
    # both inputs and outputs because inverse_* operations mutate their arguments.
    if metric_dtype == 'float64':
        batch = to_metric_dtype(batch, torch.float64)
        output = {k:v.to(dtype=torch.float64) for k,v in output.items()}
    normalized_p, normalized_q = residuals(output, batch)
    physical_output, physical = physicalize_output(output, batch, normalizer)
    p, q = residuals(physical_output, physical)
    # FP32 cross-check of the two unit-conversion routes. Record the discrepancy.
    discrepancy = torch.maximum((p - normalized_p * normalizer.baseMVA).abs().max(),
                                (q - normalized_q * normalizer.baseMVA).abs().max())
    if not torch.isfinite(discrepancy) and all(torch.isfinite(v).all() for v in output.values()):
        raise ValueError('Non-finite residuals from finite predictions')
    rows = []
    for i, sid in enumerate(batch.scenario_id.tolist()):
        nodes = batch['bus'].batch == i
        gens = batch['gen'].batch == i
        pq = nodes & batch.mask_dict['PQ']
        ref = nodes & batch.mask_dict['REF']
        nonref = nodes & ~batch.mask_dict['REF']
        if ref.sum() != 1 or not pq.any():
            raise ValueError(f'Unexpected bus masks for scenario {sid}')
        vm = (physical_output['bus'][:, VM_OUT] - physical.y_dict['bus'][:, VM_H]).abs()[pq]
        delta_va = physical_output['bus'][:, VA_OUT] - physical.y_dict['bus'][:, VA_H]
        va = (delta_va - delta_va[ref][0]).abs()[nonref]
        effective = torch.cat([p[nonref].abs(), q[pq].abs()])
        finite = bool(torch.isfinite(physical_output['bus'][nodes]).all()
                      and torch.isfinite(physical_output['gen'][gens]).all()
                      and torch.isfinite(p[nodes]).all() and torch.isfinite(q[nodes]).all())
        rows.append({
            'scenario_id': sid, 'load_scenario_idx': groups[i], 'topology_id': topologies[i],
            'split': 'official_test', 'finite': finite,
            'vm_mae_pu': vm.mean().item(), 'vm_max_pu': vm.max().item(),
            'vm_ref_mae_pu': (physical_output['bus'][ref, VM_OUT] - physical.y_dict['bus'][ref, VM_H]).abs().mean().item(),
            'va_mae_rad': va.mean().item(), 'va_max_rad': va.max().item(),
            'r_inf_pu_100MVA': effective.max().item() / 100.0,
            'p_mae_all_bus_MW': p[nodes].abs().mean().item(),
            'q_mae_all_bus_Mvar': q[nodes].abs().mean().item(),
            'pbe_mean_MVA': torch.hypot(p[nodes], q[nodes]).mean().item(),
            'pbe_max_MVA': torch.hypot(p[nodes], q[nodes]).max().item(),
        })
    return rows, discrepancy.item()


def load_model(device='cpu', model_dir=None, network=None):
    import torch
    import yaml
    from gridfm_graphkit.io.param_handler import NestedNamespace, load_normalizer, get_task
    from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
    model_dir = Path(model_dir) if model_dir is not None else ROOT / 'assets/model/case118_ieee/base/seed0'
    configs = sorted(model_dir.glob('*.yaml'))
    if len(configs) != 1:
        raise ValueError(f'expected one model config in {model_dir}, found {len(configs)}')
    cfg = yaml.safe_load(configs[0].read_text())
    configured_networks = list(cfg['data']['networks'])
    if len(configured_networks) != 1:
        raise ValueError('robustness analysis expects a single-network checkpoint')
    configured_network = str(configured_networks[0])
    if network is not None and configured_network != network:
        raise ValueError(f'checkpoint network {configured_network} does not match {network}')
    cfg['verbose'] = False
    config = NestedNamespace(**cfg)
    normalizer = load_normalizer(config)
    stats = torch.load(model_dir / 'normalizer_stats.pt', map_location='cpu', weights_only=True)
    if set(stats) != {configured_network}:
        raise ValueError('Normalizer statistics do not match the selected network')
    normalizer.fit_from_dict(stats[configured_network])
    task = get_task(config, [normalizer])
    state = torch.load(model_dir / 'best_model_state_dict.pt', map_location='cpu', weights_only=True)
    task.load_state_dict(adapt_state_dict_for_model(state, task), strict=True)
    task.to(device).eval().requires_grad_(False)
    return config, normalizer, stats, task


def model_state_sha256(task):
    """Hash checkpoint parameters and persistent buffers, excluding trace caches."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(task.state_dict().items()):
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_official_metrics(task, output, batch, normalizer, groups, topologies):
    """Compare two evaluation paths on the SAME predictions; no second forward."""
    import numpy as np
    import torch
    from unittest.mock import patch
    rows, _ = scenario_metrics(output, batch, normalizer, groups, topologies)
    logged = {}
    def capture(name, value, **kwargs):
        logged[name] = float(value)
    frozen = {k:v.clone() for k,v in output.items()}
    with patch.object(task, 'log', side_effect=capture), patch.object(
            task, 'shared_step', return_value=(frozen, {'loss':torch.tensor(0.)})):
        task.test_step(batch.clone(), 0)
    differences = {}
    for official, ours in [('Active Power Loss','p_mae_all_bus_MW'), ('Reactive Power Loss','q_mae_all_bus_Mvar'),
                           ('PBE Mean','pbe_mean_MVA'), ('PBE Max','pbe_max_MVA')]:
        actual = max(r[ours] for r in rows) if ours == 'pbe_max_MVA' else np.mean([r[ours] for r in rows])
        expected = logged['case118_ieee/' + official]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        differences[official] = float(abs(actual - expected))
    return differences


def clean_json(value):
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--limit', type=int, help='First N sorted official test IDs; default 128, 0 = all 21908. Smoke subset only.')
    selection.add_argument('--scenario-ids', type=Path, help='Explicit ID list or frozen research selection JSON; official test only')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='cpu or NVIDIA cuda')
    parser.add_argument('--batch-size', type=int,
                        help='Override training.batch_size from the official model config')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--metric-dtype', choices=['float32','float64'], default='float32', help='Metric precision; model forward stays float32')
    parser.add_argument('--verify-official', action='store_true', help='Compare first batch with official test_step using frozen FP32 predictions')
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 0) or (args.batch_size is not None and args.batch_size <= 0) or args.threads <= 0:
        parser.error('limit must be >=0; batch-size and threads must be >0')
    import numpy as np
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu or install a CUDA environment')
    lock_path = ROOT / 'assets/manifests/case118-assets.lock.json'
    lock = json.loads(lock_path.read_text())
    model_dir = ROOT / 'assets/model/case118_ieee/base/seed0'
    split_path = next((ROOT / 'assets/model/mlflow').rglob('*scenario_splits.json'), None)
    if split_path is None:
        raise FileNotFoundError('Official split missing. Run scripts/download_assets.py first.')
    model_entries = [e for e in lock['files'] if e['repo_type'] == 'model']
    for entry in model_entries:
        if not verify(ROOT / entry['local'], entry):
            raise ValueError(f"Missing/corrupt asset: {entry['local']}. Re-run download_assets.py.")
    splits = load_splits(split_path)
    metadata, selection_info = {}, None
    if args.scenario_ids:
        ids, metadata, selection_info = load_selection(args.scenario_ids, splits['test'], ROOT)
    else:
        ids = sorted(splits['test'])
        limit = 128 if args.limit is None else args.limit
        if limit:
            ids = ids[:limit]
    plan, data_entries = plan_partitions(ids, lock['files'])
    print(f'Validating {len(data_entries)} raw files for {len(ids)} official test scenarios', flush=True)
    for entry in data_entries:
        if not verify(ROOT / entry['local'], entry):
            raise ValueError(f"Missing/corrupt data: {entry['local']}. Re-run download_assets.py.")
    config, normalizer, stats, task = load_model(args.device)
    official_batch_size = int(config.training.batch_size)
    batch_size = official_batch_size if args.batch_size is None else args.batch_size
    state_before = model_state_sha256(task)
    out = args.output or ROOT / 'outputs/baseline' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'scenario_ids.json').write_text(json.dumps(ids) + '\n')
    git_commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    git_diff = subprocess.check_output(['git','diff','HEAD','--binary'], cwd=ROOT)
    (out / 'research.diff').write_bytes(git_diff)
    source_files = sorted((ROOT / 'vendor/gridfm-graphkit/gridfm_graphkit').rglob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    manifest = {
        'status': 'running', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'network': 'case118_ieee', 'model': 'Base', 'model_seed': 0,
        'graphkit_revision': lock['graphkit_revision'], 'model_revision': lock['model_revision'],
        'data_revision': lock['data_revision'], 'asset_lock_sha256': sha256(lock_path),
        'official_split_sha256': sha256(split_path), 'selected_ids_sha256': sha256(out / 'scenario_ids.json'),
        'split_counts': {k: len(v) for k, v in splits.items()},
        'selection': selection_info or ('all official test IDs' if len(ids) == len(splits['test']) else 'first N sorted official test IDs; smoke subset only'),
        'split_source': str(split_path.relative_to(ROOT)),
        'normalizer': {k: float(v) for k, v in stats['case118_ieee'].items()},
        'dtype': 'float32', 'metric_dtype':args.metric_dtype, 'tf32': False, 'compile': False, 'device': args.device,
        'run_id':out.name, 'research_git_commit':git_commit, 'research_diff_sha256':sha256(out/'research.diff'),
        'threads':args.threads,
        'model_state_sha256_before':state_before, 'parameters_require_grad':False,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'platform': platform.platform(), 'python': sys.version, 'batch_size': batch_size,
        'official_config_batch_size': official_batch_size,
        'packages': {n: importlib.metadata.version(n) for n in ('torch', 'torch-geometric', 'torch-scatter', 'numpy', 'pandas', 'pyarrow', 'lightning')},
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        'verified_files': [e['local'] for e in model_entries + data_entries],
        'note': 'Fixed topology with recorded electrical variants' if metadata else 'Official test contains topology/admittance variants.',
    }
    if args.device == 'cuda':
        manifest['gpu'] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    manifest_path = out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        records, forward_seconds, max_discrepancy = [], 0., 0.
        start = time.perf_counter()
        with (out / 'per_scenario.csv').open('w', newline='') as csvfile, torch.inference_mode():
            writer = None
            for batch_index, (batch, groups, topologies) in enumerate(graph_batches(plan, config, normalizer, batch_size, metadata)):
                batch = batch.to(args.device)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                t = time.perf_counter()
                output = task(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                forward_seconds += time.perf_counter() - t
                rows, discrepancy = scenario_metrics(output, batch, normalizer, groups, topologies, args.metric_dtype)
                for row in rows:
                    row['run_id'] = out.name
                    if metadata:
                        info = metadata[row['scenario_id']]
                        row.update({k:info[k] for k in ['research_split','electrical_id','node_map_id','cohort']})
                max_discrepancy = max(max_discrepancy, discrepancy)
                if batch_index == 0 and args.verify_official:
                    manifest['official_metric_abs_difference'] = verify_official_metrics(task, output, batch, normalizer, groups, topologies)
                    manifest['official_test_step_check'] = 'passed on first batch with frozen predictions, float32 metrics'
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                csvfile.flush()
                records.extend(rows)
                if batch_index % 20 == 0:
                    print(f'{len(records)}/{len(ids)} scenarios; forward={forward_seconds:.2f}s', flush=True)
        if sorted(r['scenario_id'] for r in records) != ids:
            raise ValueError('Evaluated IDs do not match requested IDs')
        summary = {
            'scenarios': len(records), 'load_groups': len({r['load_scenario_idx'] for r in records}),
            'topologies': len({r['topology_id'] for r in records}),
            'nonfinite_scenarios': sum(not r['finite'] for r in records),
            'forward_seconds': forward_seconds, 'evaluation_seconds': time.perf_counter() - start,
            'max_residual_unit_crosscheck_MW_or_Mvar': max_discrepancy,
        }
        # np.mean deliberately propagates nonfinite values instead of discarding failures.
        for name in ('vm_mae_pu', 'vm_ref_mae_pu', 'va_mae_rad', 'p_mae_all_bus_MW', 'q_mae_all_bus_Mvar', 'pbe_mean_MVA'):
            summary[name] = float(np.mean([r[name] for r in records]))
        rinf = [r['r_inf_pu_100MVA'] for r in records]
        summary['r_inf_pu_100MVA'] = dict(zip(('median', 'p95', 'max'), map(float, np.quantile(rinf, [.5, .95, 1]))))
        summary['pbe_max_MVA'] = float(np.max([r['pbe_max_MVA'] for r in records]))
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary['process_peak_RSS_bytes'] = int(peak_rss if sys.platform == 'darwin' else peak_rss * 1024)
        if args.device == 'cuda':
            summary['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
        summary = clean_json(summary)
        state_after = model_state_sha256(task)
        if state_after != state_before or task.training or any(p.grad is not None for p in task.parameters()):
            raise ValueError('Frozen model state changed during evaluation')
        manifest['model_state_sha256_after'] = state_after
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        manifest['status'] = 'complete'
        manifest['summary'] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print(json.dumps(summary, indent=2, allow_nan=False))
        print(f'Results: {out}')
        if summary['nonfinite_scenarios']:
            raise SystemExit('Non-finite predictions detected; see per_scenario.csv')
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = {'type':type(exc).__name__, 'message':str(exc)}
        manifest_path.write_text(json.dumps(clean_json(manifest), indent=2, allow_nan=False) + '\n')
        raise



if __name__ == '__main__':
    main()

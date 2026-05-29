#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any, Dict, Iterable, List


def iter_json(paths: Iterable[pathlib.Path]):
    for root in paths:
        if root.is_dir():
            yield from sorted(root.rglob('*.json'))
        else:
            yield root


def get(data: Dict[str, Any], path: str, default=None):
    cur: Any = data
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def flatten(path: pathlib.Path, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'path': str(path),
        'program': data.get('program'),
        'cache': data.get('cache'),
        'm': data.get('m'),
        'exact': data.get('exact'),
        'train_sec': get(data, 'training.train_sec'),
        'val_mse': get(data, 'training.val_mse'),
        'val_bias': get(data, 'training.val_bias'),
        'coarse_mean': get(data, 'estimators.coarse.stats.mean'),
        'coarse_var': get(data, 'estimators.coarse.stats.variance'),
        'coarse_bias': get(data, 'estimators.coarse.stats.bias'),
        'residual_mean': get(data, 'estimators.coupled_residual.residual_stats.mean'),
        'residual_var': get(data, 'estimators.coupled_residual.residual_stats.variance'),
        'residual_var_ratio_vs_pure': get(data, 'two_level.residual_var_ratio_vs_pure'),
        'two_level_mean': get(data, 'two_level.mean'),
        'two_level_bias': get(data, 'two_level.bias'),
        'two_level_mse': get(data, 'two_level.mse_model_current_allocation'),
        'speedup_score_vs_pure': get(data, 'two_level.speedup_score_vs_pure'),
        'coarse_infer_usec_per_sample': get(data, 'timing.coarse_inference_usec_per_sample'),
        'residual_infer_usec_per_sample': get(data, 'timing.residual_inference_usec_per_sample'),
        'coarse_total_usec_per_sample': get(data, 'timing.coarse_total_usec_per_sample'),
        'residual_total_usec_per_sample': get(data, 'timing.residual_total_usec_per_sample'),
        'pure_usec_per_sample': get(data, 'timing.pure_usec_per_sample'),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=pathlib.Path)
    parser.add_argument('-o', '--out', type=pathlib.Path, default=None)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for p in iter_json(args.paths):
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get('program') != 'n2wos_tcnn_2lmc_postprocess':
            continue
        rows.append(flatten(p, data))

    if not rows:
        raise SystemExit('no n2wos_tcnn_2lmc_postprocess JSON files found')

    fields = list(rows[0].keys())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        f = args.out.open('w', newline='', encoding='utf-8')
    else:
        import sys
        f = sys.stdout
    with f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Render complete server matrix summaries without reading raster pixels.

Run after sync_server_upgrade_results.py. All three seeds and both evaluation
roles are required. Seed ranges in the display are not confidence intervals;
the frozen process intervals remain in the source CSV and primary JSON.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'outputs/server_migration_20260929/remote_results'
DEFAULT_CONFIG = ROOT / 'outputs/server_migration_20260929/package/configs/upgrade_20260929.json'
DEFAULT_OUTPUT = ROOT / 'outputs/server_migration_20260929/publication'
ARM = {('B5_whole', 'original'): 'A', ('B5_whole', 'multistate'): 'B',
       ('B5_local', 'original'): 'C', ('B5_local', 'multistate'): 'D'}
METRICS = ('binary_flood_brier', 'flood_iou', 'flood_f1', 'flood_auprc',
           'multiclass_nll', 'multiclass_brier', 'ece15')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_complete(source, config_path):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    base = source / 'outputs/upgrade_20260929'
    summary = base / 'summaries'
    marker = summary / 'matrix_summary_audit.json'
    if not marker.is_file():
        raise FileNotFoundError('Full matrix summary is not ready; no publication artifacts created.')
    record = json.loads(marker.read_text(encoding='utf-8'))
    if (record.get('status') != 'COMPLETE' or not record.get('roles_separate')
            or record.get('interpretation') != 'retrospective descriptive only'
            or record.get('evaluation_record_count') != {'internal_retro': 24, 'holdout_retro': 24}):
        raise ValueError('Matrix completion record does not cover the frozen 48 evaluations.')
    hashes = {}
    for section in ('outputs', 'input_file_hashes', 'B1_hard_only_file_hashes'):
        for relative, expected in record[section].items():
            path = source / relative
            observed = sha(path)
            if observed != expected:
                raise ValueError('Synced artifact differs from matrix summary: ' + relative)
            hashes[relative] = observed
    expected_config = sha(config_path)
    if record['primary_interaction']['config_sha256'] != expected_config:
        raise ValueError('Publication config differs from evaluated config.')
    for model, spec in config['models'].items():
        for seed in config['seeds']:
            folder = base / f'{model}_seed_{seed}'
            run = json.loads((folder / 'run.json').read_text())
            epochs = [json.loads(x) for x in (folder / 'epochs.jsonl').read_text().splitlines() if x]
            if (run['status'] != 'FIT_COMPLETE' or run['completed_epochs'] != config['epochs']
                    or len(epochs) != config['epochs'] or run['hashes']['config'] != expected_config):
                raise ValueError('Missing full frozen training budget: ' + folder.name)
            for strategy in spec['calibration']:
                temp = json.loads((folder / f'temperature_{strategy}.json').read_text())
                if (temp['config_sha256'] != expected_config
                        or not math.isfinite(temp['temperature']) or temp['temperature'] <= 0):
                    raise ValueError('Invalid global temperature: ' + folder.name)
                for role in ('internal_retro', 'holdout_retro'):
                    result = json.loads((folder / f'evaluation_{role}_{strategy}.json').read_text())
                    if result.get('config_sha256') != expected_config:
                        raise ValueError('Evaluation config mismatch: ' + folder.name)
    rows = {}
    for role, filename in (('internal_retro', 'internal_metric_summary.csv'),
                           ('holdout_retro', 'holdout_stratified_summary.csv')):
        with (summary / filename).open(newline='', encoding='utf-8') as stream:
            rows[role] = list(csv.DictReader(stream))
    return config, record, rows, hashes


def aggregate(rows, seeds):
    """Keep roles/strata/states separate; show each seed and its equal mean."""
    groups = {}
    for row in rows:
        key = tuple(row.get(k, '') for k in ('model', 'calibration_strategy', 'stratum', 'state', 'score'))
        cells = groups.setdefault(key, {})
        if row['seed'] in cells:
            raise ValueError('Duplicate seed summary cell: ' + str(key))
        cells[row['seed']] = row
    result = []
    for key, cells in sorted(groups.items()):
        if set(cells) != {str(s) for s in seeds}:
            raise ValueError('Missing fixed seeds in summary cell: ' + str(key))
        row = dict(zip(('model', 'calibration_strategy', 'stratum', 'state', 'score'), key))
        row['arm'] = ARM.get(key[:2], '')
        for metric in METRICS:
            values = [cells[str(s)][metric + '_mean'] for s in seeds]
            numbers = [float(v) for v in values if v not in ('', 'None', 'nan')]
            # Undefined seeds remain undefined; never silently average a subset.
            complete = len(numbers) == len(seeds) and all(map(math.isfinite, numbers))
            row[metric + '_mean'] = sum(numbers) / len(numbers) if complete else None
            row[metric + '_seed_min'] = min(numbers) if complete else None
            row[metric + '_seed_max'] = max(numbers) if complete else None
            row[metric + '_process_n_min'] = min(int(cells[str(s)][metric + '_n']) for s in seeds)
            for seed, value in zip(seeds, values):
                row[f'{metric}_seed_{seed}'] = value
        result.append(row)
    return result


def display(value):
    return '--' if value is None else f'{value:.5f}'


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def runtime_and_temperatures(source, config, output):
    base = source / 'outputs/upgrade_20260929'
    registry = base / 'orchestrator/runs.jsonl'
    events = [json.loads(x) for x in registry.read_text().splitlines() if x]
    costs = []
    for event in events:
        if event.get('event') != 'FINISH':
            continue
        costs.append({key: event.get(key) for key in (
            'job_id', 'status', 'device', 'exit_code', 'started_utc', 'finished_utc',
            'elapsed_accelerator_hours', 'stop_reason', 'source_run_sha256')})
    write_csv(output / 'registered_runtime.csv', costs)
    temperatures, curves = [], []
    extra_hashes = {registry.relative_to(source).as_posix(): sha(registry)}
    for model, spec in config['models'].items():
        for seed in config['seeds']:
            folder = base / f'{model}_seed_{seed}'
            epochs_path = folder / 'epochs.jsonl'
            extra_hashes[epochs_path.relative_to(source).as_posix()] = sha(epochs_path)
            for line in epochs_path.read_text().splitlines():
                if line:
                    row = json.loads(line)
                    curves.append({key: row.get(key) for key in (
                        'run_id', 'model', 'seed', 'epoch', 'learning_rate',
                        'mean_batch_loss', 'duration_seconds', 'batches', 'train_tiles',
                        'non_ignore_pixels', 'checkpoint_sha256')})
            for strategy in spec['calibration']:
                record = json.loads((folder / f'temperature_{strategy}.json').read_text())
                temperatures.append({'model': model, 'seed': seed, 'strategy': strategy,
                    'arm': ARM.get((model, strategy), ''),
                    'temperature': record['temperature'],
                    'objective_nll': record['objective_nll'],
                    'boundary_hit': record['boundary_hit'],
                    'calibration_processes': len(record['process_ids']),
                    'calibration_states': '|'.join(record['state_ids']),
                    'checkpoint_sha256': record['checkpoint_sha256']})
    write_csv(output / 'temperatures.csv', temperatures)
    write_csv(output / 'training_curves.csv', curves)
    return extra_hashes


def label(row):
    if row['arm']:
        return row['arm']
    return row['model']


def render_latex(rows, primary, output):
    """Generate an editable table fragment; do not alter the manuscript yet."""
    cells = {(label(row), row['state']): row for row in rows
             if row['score'] == 'calibrated'}
    order = ('B2', 'B4', 'P', 'A', 'B', 'C', 'D', 'SMAGNet')
    lines = [r'\begin{table}[H]',
        r'\caption{Frozen upgrade, internal retrospective evaluation. '
        r'Entries are equal means of three fixed seeds after process-level scoring. '
        r'Columns show 100\% and 0\% optical retention; all intermediate and '
        r'irregular states, seed values, supports and process intervals remain '
        r'in the accompanying CSVs. Flood Brier uses calibrated probabilities. '
        r'Positive temperature preserves three-class argmax.}',
        r'\label{tab:upgrade20260929}', r'\centering\normalsize',
        r'\renewcommand{\arraystretch}{1.16}',
        r'\begin{tabular*}{0.96\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}', r'\hline',
        r'Model/arm & Brier 100\% & Brier 0\% & Flood IoU 100\% & Flood IoU 0\% \\',
        r'\hline']
    for name in order:
        values = [cells[(name, state)][metric + '_mean']
                  for metric in ('binary_flood_brier', 'flood_iou') for state in ('100', '0')]
        lines.append(name + ' & ' + ' & '.join(display(v) for v in values) + r' \\')
    with (output / 'b1_hard_metrics.csv').open(newline='', encoding='utf-8') as stream:
        b1_rows = [r for r in csv.DictReader(stream) if r['role'] == 'internal_retro'
                   and r['process_id'] == 'PROCESS_MACRO' and r['metric'] == 'flood_iou']
    if len(b1_rows) != 1:
        raise ValueError('Exactly one internal B1 process-macro Flood IoU required.')
    b1_iou = float(b1_rows[0]['mean']) if b1_rows[0]['mean'] else None
    lines.append('B1 (SAR rule) & -- & -- & ' + display(b1_iou) + ' & '
                 + display(b1_iou) + r' \\')
    lines += [r'\hline', r'\end{tabular*}',
              r'\par\smallskip\raggedright\small A/B: whole-modality dropout; '
              r'C/D: mixed local and whole dropout. A/C: original-state temperature; '
              r'B/D: multistate temperature. B1 is a fixed SAR-only hard-label rule '
              r'with no probability metrics; its result is identical across '
              r'optical interventions. This comparison is descriptive and '
              r'does not establish independent generalization.', r'\end{table}', '']
    (output / 'upgrade_internal_table.tex').write_text('\n'.join(lines), encoding='utf-8')
    lines = [r'\begin{table}[H]',
             r'\caption{Prespecified flood-Brier interaction $(D-C)-(B-A)$, '
             r'five-state equal mean within process. The overall interval is '
             r'the frozen paired process bootstrap with seeds retained jointly '
             r'(10,000 resamples). Seed rows are point estimates.}',
             r'\label{tab:upgradeprimary}', r'\centering\normalsize',
             r'\renewcommand{\arraystretch}{1.22}',
             r'\begin{tabular*}{0.80\linewidth}{@{\extracolsep{\fill}}lcc@{}}',
             r'\toprule',
             r'Seed/aggregation & Interaction & 95\% process interval \\', r'\midrule']
    for seed, value in primary['per_seed'].items():
        lines.append(seed + ' & ' + display(value) + r' & -- \\')
    interval = primary.get('ci95')
    ci_text = '--' if interval is None else '[' + ', '.join(display(v) for v in interval) + ']'
    lines.append('Equal seed mean & ' + display(primary['mean']) + ' & ' + ci_text + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}', '']
    (output / 'upgrade_primary_table.tex').write_text('\n'.join(lines), encoding='utf-8')


def paired_flood_iou_guardrail(evaluations, seeds, process_ids, bootstrap, tolerance):
    """Frozen C-A at 100%; undefined pairs retain their explicit support.

    Resampling uses one process vector for all seeds. If a resample has no
    defined pairs for a seed, the interval is undefined rather than silently
    dropping that resample or changing its denominator.
    """
    values = np.full((len(process_ids), len(seeds)), np.nan, dtype=np.float64)
    excluded = {}
    for si, seed in enumerate(seeds):
        for pi, process in enumerate(process_ids):
            pair = []
            for model in ('B5_whole', 'B5_local'):
                rows = evaluations[(model, seed)]['per_process']
                prefix = f'{model}|{seed}|original|{process}|100|'
                raw = rows[prefix + 'raw']['flood_iou']
                calibrated = rows[prefix + 'calibrated']['flood_iou']
                if raw != calibrated:
                    raise ValueError('Global temperature changed Flood IoU: ' + prefix)
                if raw is not None and (not math.isfinite(raw) or not 0 <= raw <= 1):
                    raise ValueError('Invalid Flood IoU: ' + prefix)
                pair.append(raw)
            if any(v is None for v in pair):
                excluded.setdefault(str(seed), []).append(process)
            else:
                values[pi, si] = pair[1] - pair[0]
    support = np.isfinite(values)
    counts = support.sum(axis=0)
    report = {'contrast': 'C-A at 100% optical availability', 'metric': 'flood_iou',
              'tolerance': tolerance, 'interpretation': 'project-defined descriptive tolerance; not confirmatory noninferiority',
              'process_ids': list(process_ids), 'excluded_pairs_by_seed': excluded,
              'per_seed': {str(s): {'n_processes': int(counts[i]),
                  'mean': float(values[support[:, i], i].mean()) if counts[i] else None}
                  for i, s in enumerate(seeds)},
              'bootstrap': dict(bootstrap), 'common_pair_rule': 'A/C both defined within each seed'}
    if np.any(counts == 0):
        report.update(status='undefined_no_seed_support', mean=None, ci95=None,
                      one_sided_95_lower=None, compatible_with_tolerance=None)
        return report
    point_by_seed = np.nansum(values, axis=0) / counts
    rng = np.random.Generator(np.random.PCG64(bootstrap['seed']))
    repetitions = bootstrap['replicates']
    boot = np.empty(repetitions)
    empty_draws = 0
    for ri in range(repetitions):
        draw = rng.integers(0, len(process_ids), size=len(process_ids))
        sample = values[draw]
        draw_counts = np.isfinite(sample).sum(axis=0)
        if np.any(draw_counts == 0):
            empty_draws += 1
            boot[ri] = np.nan
        else:
            boot[ri] = (np.nansum(sample, axis=0) / draw_counts).mean()
    report.update(mean=float(point_by_seed.mean()),
                  paired_process_effects={p: {str(s): float(values[i,j]) if support[i,j] else None
                      for j,s in enumerate(seeds)} for i,p in enumerate(process_ids)},
                  bootstrap_empty_seed_support_draws=empty_draws)
    if empty_draws:
        report.update(status='undefined_bootstrap_empty_seed_support', ci95=None,
                      one_sided_95_lower=None, compatible_with_tolerance=None)
    else:
        low, high = np.quantile(boot, [.025, .975])
        lower = float(np.quantile(boot, .05))
        report.update(status='defined', ci95=[float(low),float(high)],
                      one_sided_95_lower=lower, compatible_with_tolerance=lower >= tolerance,
                      boundary_sensitivity={str(t): lower >= t for t in (-.01, -.03)})
    return report


def render(source, config_path, output):
    config, evidence, roles, hashes = load_complete(source, config_path)
    aggregates = {role: aggregate(rows, config['seeds']) for role, rows in roles.items()}
    # Only now create output files; an unfinished experiment gets no fake table.
    output.mkdir(parents=True, exist_ok=True)
    hashes.update(runtime_and_temperatures(source, config, output))
    for name in ('b1_hard_metrics.csv', 'internal_metric_summary.csv',
                 'holdout_stratified_summary.csv', 'primary_interaction_internal_retro.json'):
        shutil.copy2(source / 'outputs/upgrade_20260929/summaries' / name, output / name)
    for role, rows in aggregates.items():
        with (output / f'{role}_all_seeds.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Frozen experiment result displays', '',
        'Retrospective descriptive results; no new blind validation or independent review PASS.', '',
        'All seeds are retained. Displayed means weight seeds equally after process-level scoring. '
        'Seed ranges are not confidence intervals; source CSVs retain process-bootstrap intervals.', '',
        'A: whole dropout/original calibration; B: whole dropout/multistate calibration; '
        'C: mixed local/whole dropout/original calibration; D: mixed local/whole dropout/multistate calibration.', '']
    for role, rows in aggregates.items():
        lines += ['## ' + role, '',
                  '| Model/arm | Stratum | State | Flood Brier | Flood IoU | Flood F1 | Flood AUPRC |',
                  '|---|---|---|---:|---:|---:|---:|']
        for row in rows:
            if row['score'] != 'calibrated':
                continue
            values = [display(row[x + '_mean']) for x in METRICS[:4]]
            lines.append('| ' + ' | '.join([label(row), row['stratum'] or 'internal', row['state'], *values]) + ' |')
        lines.append('')
    primary = evidence['primary_interaction']
    evals = {(m, s): json.loads((source / 'outputs/upgrade_20260929' / f'{m}_seed_{s}'
                               / 'evaluation_internal_retro_original.json').read_text())
             for m in ('B5_whole', 'B5_local') for s in config['seeds']}
    guardrail = paired_flood_iou_guardrail(evals, config['seeds'], primary['expected_process_ids'],
        config['evaluation_protocol']['bootstrap'],
        config['evaluation_protocol']['flood_iou_guardrail']['tolerance'])
    (output / 'flood_iou_guardrail.json').write_text(json.dumps(guardrail, indent=2) + '\n', encoding='utf-8')
    render_latex(aggregates['internal_retro'], primary, output)
    lines += ['## Prespecified primary contrast', '',
              'Flood Brier interaction (D-C)-(B-A), five states equally weighted within process; '
              'paired process bootstrap preserves arms and seeds.', '',
              f"Status: {primary['status']}; mean: {display(primary['mean'])}; "
              f"95% interval: {primary.get('ci95')}; seed effects: {primary['per_seed']}", '',
              'A negative interaction means multistate calibration gives a greater Brier reduction '
              'under the mixed training strategy relative to whole dropout. '
              'It does not by itself establish superiority against all baselines.', '',
              '## Frozen segmentation guardrail', '',
              f"C-A at 100% optical: {guardrail['mean']}; one-sided 95% lower: "
              f"{guardrail['one_sided_95_lower']}; project tolerance: {guardrail['tolerance']}; "
              f"compatible: {guardrail['compatible_with_tolerance']}; status: {guardrail['status']}", '',
              'Undefined pairs, support per seed and all process effects remain in flood_iou_guardrail.json. '
              'Compatibility is descriptive and does not establish confirmatory noninferiority.', '',
              'Positive global temperature preserves three-class argmax. B1 has hard-label segmentation '
              'metrics only and is excluded from probability plots; see the separate b1_hard_metrics.csv.', '',
              'Runtime CSV records all registered FINISH attempts and retains null exit codes for '
              'imported completed XPU weights. Elapsed time includes I/O and initialization; '
              'it is not a pure inference benchmark. Temperatures and all 24-epoch loss curves '
              'are exported separately. Training loss is not test performance.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    states = config['evaluation_protocol']['primary_states']
    for metric, title in (('binary_flood_brier', 'Flood binary Brier (lower is better)'),
                          ('flood_iou', 'Flood IoU (higher is better)')):
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        for arm in 'ABCD':
            selected = {r['state']: r for r in aggregates['internal_retro']
                        if r['arm'] == arm and r['score'] == 'calibrated'}
            values = [selected[s][metric + '_mean'] for s in states]
            ax.plot(range(len(states)), values, marker='o', label=arm, linewidth=1.3)
            for seed in config['seeds']:
                seed_values = [selected[s][f'{metric}_seed_{seed}'] for s in states]
                ax.scatter(range(len(states)),
                           [float(v) if v not in ('', 'None', 'nan') else math.nan for v in seed_values],
                           s=10, alpha=.4, color=ax.lines[-1].get_color())
        ax.set_xticks(range(len(states)), [s + '%' for s in states])
        ax.set_xlabel('Retained originally valid optical area')
        ax.set_ylabel(title)
        ax.set_title('Internal retrospective evaluation; line = mean of three seeds')
        ax.grid(alpha=.2)
        ax.legend(title='Frozen arm', ncol=4, frameon=False)
        fig.tight_layout()
        for suffix in ('pdf', 'svg', 'png'):
            fig.savefig(output / f'b5_{metric}.{suffix}', dpi=180)
        plt.close(fig)
    provenance = {'status': 'COMPLETE', 'source': str(source), 'config_sha256': sha(config_path),
                  'interpretation': 'retrospective descriptive only',
                  'seed_ranges_are_not_confidence_intervals': True,
                  'source_artifact_hashes': hashes, 'generator_sha256': sha(Path(__file__)),
                  'generated_files': {p.name: sha(p) for p in output.iterdir()
                                      if p.is_file() and p.name != 'provenance.json'}}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        render(args.source, args.config, args.output)
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(2)


if __name__ == '__main__':
    main()

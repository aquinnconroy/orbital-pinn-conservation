#!/usr/bin/env python3
"""
Figure Generation for PINN N-Body Conservation Paper

Generates all 8 figures for the paper from experiment data:
  - Figures 1-6: Main paper figures from paper_runs_final.json
  - Figure A1:   Appendix leapfrog drift comparison from appendix_drift_data.json
  - Figure A2:   Appendix hyperparameter correlation heatmap

Usage:
    python generate_figures.py
    python generate_figures.py --outdir figures/

Input data (in data/ directory):
    paper_runs_final.json        92 FP64 runs from W&B (28 baseline + 34 all_three + 30 ablation)
    rollout_length_comparison.csv  Multi-horizon rollout results (top-5 per group)
    appendix_drift_data.json     Leapfrog integrator drift at 5 timestep scales

Author: Quinn Conroy
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Plot style (JEI submission: Arial 11pt) ──────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'baseline': '#888888',
    'conservation': '#2166AC',
    'energy': '#D6604D',
    'momentum': '#4393C3',
    'angular': '#762A83',
    'all_three': '#2166AC',
}


def load_runs(path):
    """Load and group paper runs."""
    with open(path) as f:
        data = json.load(f)
    runs = data['runs'] if 'runs' in data else data
    groups = {}
    for r in runs:
        groups.setdefault(r['category'], []).append(r)
    return runs, groups


def get_field(runs, field):
    """Extract non-null values of a field."""
    return [r[field] for r in runs if r.get(field) is not None]


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Validation MSE Comparison
# ══════════════════════════════════════════════════════════════════════
def fig1_mse_comparison(groups, outdir):
    bl_mse = sorted(get_field(groups['baseline'], 'best_val_mse'))
    cn_mse = sorted(get_field(groups['all_three'], 'best_val_mse'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), gridspec_kw={'width_ratios': [1, 1.6]})

    means = [np.mean(bl_mse), np.mean(cn_mse)]
    stds = [np.std(bl_mse), np.std(cn_mse)]
    medians = [np.median(bl_mse), np.median(cn_mse)]

    ax1.bar(['Baseline\n(MSE only)', 'Conservation\n(MSE + physics)'],
            means, yerr=stds, capsize=5, color=[COLORS['baseline'], COLORS['conservation']],
            edgecolor='black', linewidth=0.5, width=0.55, zorder=3)
    ax1.scatter([0, 1], medians, marker='D', color='white', edgecolor='black', s=30, zorder=4, label='Median')
    ax1.set_ylabel('Best Validation MSE')
    ax1.legend(loc='upper right', framealpha=0.8)

    pct = (means[0] - means[1]) / means[0] * 100
    ax1.annotate(f'{pct:.1f}% lower\nmean MSE',
                xy=(1, means[1] + stds[1] + 100), fontsize=8,
                ha='center', color=COLORS['conservation'], fontweight='bold')

    ax2.plot(range(len(bl_mse)), bl_mse, 'o-', color=COLORS['baseline'],
             markersize=3, linewidth=1, label=f'Baseline (n={len(bl_mse)})', alpha=0.8)
    ax2.plot(range(len(cn_mse)), cn_mse, 's-', color=COLORS['conservation'],
             markersize=3, linewidth=1, label=f'Conservation (n={len(cn_mse)})', alpha=0.8)
    ax2.set_xlabel('Run Index (sorted by MSE)')
    ax2.set_ylabel('Best Validation MSE')
    ax2.legend(loc='upper left', framealpha=0.8)

    fig.tight_layout()
    path = outdir / 'fig1_mse_comparison.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  Fig 1: {path.name} | BL n={len(bl_mse)}, CN n={len(cn_mse)}')
    return path


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Conservation Drift Box Plots
# ══════════════════════════════════════════════════════════════════════
def fig2_conservation_drift(groups, outdir):
    bl = groups['baseline']
    cn = groups['all_three']
    n_bl, n_cn = len(bl), len(cn)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.8))

    metrics = [
        ('Energy Drift', 'energy_drift', COLORS['energy']),
        ('Momentum Drift', 'momentum_drift', COLORS['momentum']),
        ('Ang. Mom. Drift', 'angular_momentum_drift', COLORS['angular']),
    ]

    for ax, (title, field, color) in zip(axes, metrics):
        bl_data = [r[field] for r in bl if r.get(field) is not None]
        cn_data = [r[field] for r in cn if r.get(field) is not None]

        bl_clean = [x for x in bl_data if x < 1e8]
        cn_clean = [x for x in cn_data if x < 1e8]

        bp = ax.boxplot([bl_clean, cn_clean], tick_labels=['Baseline', 'Conserv.'],
                       patch_artist=True, widths=0.5, showfliers=True,
                       flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.4})
        bp['boxes'][0].set_facecolor(COLORS['baseline']); bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(color); bp['boxes'][1].set_alpha(0.6)
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black', linewidth=0.8)

        ax.set_yscale('log')
        if ax == axes[0]:
            ax.set_ylabel('Drift (log scale)')

        bl_med = np.median(bl_data)
        cn_med = np.median(cn_data)
        if cn_med > 0:
            ratio = bl_med / cn_med
            if ratio > 1e4:
                label = f'~{ratio:.2g}x'
            else:
                label = f'{ratio:.0f}x'
            ax.text(0.5, 0.92, label, transform=ax.transAxes, fontsize=8, ha='center',
                   color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9))

    fig.tight_layout()
    path = outdir / 'fig2_conservation_drift.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  Fig 2: {path.name} | BL n={n_bl}, CN n={n_cn}')
    return path


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Parallel Coordinates
# ══════════════════════════════════════════════════════════════════════
def fig3_parallel_coordinates(groups, outdir):
    cn = groups['all_three']
    data = []
    for r in cn:
        mse = r.get('best_val_mse')
        if mse is None: continue
        data.append({
            'w_energy': r.get('weight_energy', 0),
            'w_momentum': r.get('weight_momentum', 0),
            'w_angular': r.get('weight_angular_momentum', 0),
            'lr': r.get('learning_rate', 0),
            'mse': mse,
        })

    if not data:
        print('  Fig 3: No data!')
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    dims = ['w_energy', 'w_momentum', 'w_angular', 'lr', 'mse']
    dim_labels = ['Weight Energy', 'Weight Momentum', 'Weight Ang. Mom.', 'Learning Rate', 'Val MSE']

    vals = {d: [x[d] for x in data] for d in dims}
    mins = {d: min(vals[d]) for d in dims}
    maxs = {d: max(vals[d]) for d in dims}

    x_positions = np.arange(len(dims))
    mse_vals = np.array([x['mse'] for x in data])
    mse_norm = (mse_vals - mse_vals.min()) / (mse_vals.max() - mse_vals.min() + 1e-9)
    cmap = plt.cm.RdYlBu_r

    for i, row in enumerate(data):
        y_vals = []
        for d in dims:
            rng = maxs[d] - mins[d]
            y_vals.append((row[d] - mins[d]) / rng if rng > 1e-12 else 0.5)
        ax.plot(x_positions, y_vals, color=cmap(mse_norm[i]),
                alpha=0.3 + 0.5 * (1 - mse_norm[i]), linewidth=0.8)

    for x in x_positions:
        ax.axvline(x, color='black', linewidth=0.5, alpha=0.3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(dim_labels, fontsize=9)
    ax.set_ylabel('Normalized Value')

    for i, d in enumerate(dims):
        ax.text(i, -0.12, f'{mins[d]:.3g}', ha='center', fontsize=7, color='gray')
        ax.text(i, 1.08, f'{maxs[d]:.3g}', ha='center', fontsize=7, color='gray')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(mse_vals.min(), mse_vals.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Val MSE', fontsize=8)
    ax.set_ylim(-0.2, 1.2)

    fig.tight_layout()
    path = outdir / 'fig3_parallel_coordinates.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  Fig 3: {path.name} | n={len(data)} conservation runs')
    return path


# ══════════════════════════════════════════════════════════════════════
# Figure 4: 500-Step Comparison
# ══════════════════════════════════════════════════════════════════════
def fig4_500step_comparison(groups, outdir):
    bl_500 = sorted(get_field(groups['baseline'], 'mean_pos_mse_500step'))
    cn_500 = sorted(get_field(groups['all_three'], 'mean_pos_mse_500step'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5), gridspec_kw={'width_ratios': [1.2, 1]})

    ax1.scatter(range(len(bl_500)), bl_500, c=COLORS['baseline'], s=15, alpha=0.7,
                label=f'Baseline (n={len(bl_500)})', zorder=3)
    ax1.scatter(range(len(cn_500)), cn_500, c=COLORS['conservation'], s=15, alpha=0.7,
                label=f'Conservation (n={len(cn_500)})', zorder=3)
    ax1.set_yscale('log')
    ax1.set_xlabel('Run Index (sorted)')
    ax1.set_ylabel('500-Step Position MSE (log)')
    ax1.legend(loc='upper left', framealpha=0.8)

    ax2.scatter(range(len(cn_500)), cn_500, c=COLORS['conservation'], s=20, alpha=0.8, zorder=3)
    ax2.set_xlabel('Run Index (sorted)')
    ax2.set_ylabel('500-Step Position MSE')
    cn_mean, cn_std = np.mean(cn_500), np.std(cn_500)
    ax2.axhline(cn_mean, color=COLORS['conservation'], linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = outdir / 'fig4_500step_comparison.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  Fig 4: {path.name} | BL n={len(bl_500)}, CN n={len(cn_500)}')
    return path


# ══════════════════════════════════════════════════════════════════════
# Figure 5: Error Growth Analysis
# ══════════════════════════════════════════════════════════════════════
def fig5_error_growth(groups, outdir, rollout_csv=None):
    if rollout_csv is None or not rollout_csv.exists():
        print('  Fig 5: No rollout CSV, skipping')
        return None

    df = pd.read_csv(rollout_csv)
    bl_df = df[df['type'] == 'baseline']
    cn_df = df[df['type'] == 'conservation']

    horizons = sorted(bl_df['rollout_length'].unique())
    bl_means, cn_means = [], []
    for h in horizons:
        bl_means.append(bl_df[bl_df['rollout_length'] == h]['mean_pos_mse'].mean())
        cn_means.append(cn_df[cn_df['rollout_length'] == h]['mean_pos_mse'].mean())

    bl_means, cn_means = np.array(bl_means), np.array(cn_means)
    horizons = np.array(horizons)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    ax1.loglog(horizons, bl_means, 'o-', color=COLORS['baseline'], markersize=5, linewidth=1.5, label='Baseline (top-5 avg)')
    ax1.loglog(horizons, cn_means, 's-', color=COLORS['conservation'], markersize=5, linewidth=1.5, label='Conservation (top-5 avg)')

    valid = (bl_means > 0) & (cn_means > 0)
    if np.sum(valid) > 2:
        bl_fit = np.polyfit(np.log10(horizons[valid]), np.log10(bl_means[valid]), 1)
        cn_fit = np.polyfit(np.log10(horizons[valid]), np.log10(cn_means[valid]), 1)
        ax1.text(0.05, 0.95, f'Baseline: ~t^{bl_fit[0]:.2f}', transform=ax1.transAxes,
                fontsize=8, va='top', color=COLORS['baseline'], fontweight='bold')
        ax1.text(0.05, 0.87, f'Conservation: ~t^{cn_fit[0]:.2f}', transform=ax1.transAxes,
                fontsize=8, va='top', color=COLORS['conservation'], fontweight='bold')

    ax1.set_xlabel('Rollout Steps')
    ax1.set_ylabel('Position MSE')
    ax1.legend(loc='lower right', framealpha=0.8, fontsize=8)

    ratio = bl_means / cn_means
    ax2.plot(horizons, ratio, 'o-', color='#27AE60', markersize=6, linewidth=2)
    for h, r in zip(horizons, ratio):
        ax2.text(h, r * 1.1, f'{r:.1f}x', ha='center', fontsize=7, fontweight='bold')
    ax2.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Rollout Steps')
    ax2.set_ylabel('Baseline / Conservation MSE')
    ax2.set_xscale('log')

    fig.tight_layout()
    path = outdir / 'fig5_error_growth.png'
    fig.savefig(path)
    plt.close(fig)
    n_bl = bl_df['run_id'].nunique()
    n_cn = cn_df['run_id'].nunique()
    print(f'  Fig 5: {path.name} | BL top-{n_bl}, CN top-{n_cn}')
    return path


# ══════════════════════════════════════════════════════════════════════
# Figure 6: Ablation Study
# ══════════════════════════════════════════════════════════════════════
def fig6_ablation(groups, outdir):
    configs = [
        ('Baseline', groups.get('baseline', []), COLORS['baseline']),
        ('Energy\nOnly', groups.get('energy_only', []), COLORS['energy']),
        ('Momentum\nOnly', groups.get('momentum_only', []), COLORS['momentum']),
        ('Ang. Mom.\nOnly', groups.get('angular_only', []), COLORS['angular']),
        ('All Three', groups.get('all_three', []), COLORS['all_three']),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (label, runs, color) in enumerate(configs):
        mses = get_field(runs, 'mean_pos_mse_500step')
        if not mses: continue

        log_mses = np.log10(np.array(mses))
        mean_log = np.mean(log_mses)
        std_log = np.std(log_mses)

        ax.bar(i, 10**mean_log, color=color, alpha=0.7, edgecolor='black', linewidth=0.5, width=0.6)
        ax.errorbar(i, 10**mean_log,
                    yerr=[[10**mean_log - 10**(mean_log - std_log)], [10**(mean_log + std_log) - 10**mean_log]],
                    fmt='none', color='black', capsize=4, linewidth=1)
        ax.text(i, 10**(mean_log + std_log + 0.05), f'n={len(mses)}', ha='center', fontsize=8, fontweight='bold')

    ax.set_yscale('log')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c[0] for c in configs], fontsize=9)
    ax.set_ylabel('500-Step Position MSE (log scale)')

    fig.tight_layout(pad=1.5)
    path = outdir / 'fig6_ablation.png'
    fig.savefig(path)
    plt.close(fig)

    for label, runs_list, _ in configs:
        mses = get_field(runs_list, 'mean_pos_mse_500step')
        print(f'  {label.replace(chr(10)," ")}: n={len(mses)}, mean={np.mean(mses):.1f}')
    print(f'  Fig 6: {path.name}')
    return path


# ══════════════════════════════════════════════════════════════════════
# Figure A1: Leapfrog Integrator Drift Comparison
# ══════════════════════════════════════════════════════════════════════
def fig_a1_drift(outdir, drift_data_path):
    with open(drift_data_path) as f:
        data = json.load(f)

    multipliers = data['multipliers']
    n_steps = data['params']['n_steps']
    dt_nom = data['params']['dt_nominal']
    steps = np.arange(1, n_steps + 1)
    time = steps * dt_nom

    colors = {0.25: '#1a9641', 0.5: '#a6d96a', 1.0: '#404040', 2.0: '#fdae61', 4.0: '#d7191c'}
    labels = {
        0.25: '0.25x dt', 0.5: '0.5x dt', 1.0: '1x dt (nominal)',
        2.0: '2x dt', 4.0: '4x dt',
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    panel_info = [
        ('Energy Drift', 'E', r'Relative energy drift $|\Delta E|/|E_0|$'),
        ('Linear Momentum Drift', 'P', r'Relative momentum drift $|\Delta \mathbf{P}|/P_{char}$'),
        ('Angular Momentum Drift', 'L', r'Relative ang. mom. drift $|\Delta \mathbf{L}|/L_{char}$'),
    ]

    for ax, (title, key, ylabel) in zip(axes, panel_info):
        for mult in multipliers:
            drift = np.array(data['results'][str(mult)][key])
            color = colors[mult]
            lw = 2.0 if mult == 1.0 else 1.5
            ls = '-' if mult == 1.0 else ('--' if mult < 1.0 else ':')
            ax.semilogy(time, drift, color=color, label=labels[mult],
                        linewidth=lw, linestyle=ls, alpha=0.9)

        ax.set_xlabel('Simulation time')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, n_steps * dt_nom)
        ax.grid(True, alpha=0.3, linestyle='--')

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='upper center', ncol=5, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02), fontsize=12)

    fig.tight_layout()

    path = outdir / 'fig_a1_drift_comparison.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig A1: {path.name}')
    return path


# ══════════════════════════════════════════════════════════════════════
# Figure A2: Hyperparameter Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════
def fig_a2_correlation_heatmap(groups, outdir):
    conservation = groups['all_three']

    param_keys = [
        'weight_energy', 'weight_momentum', 'weight_angular_momentum',
        'learning_rate', 'temporal_decay_alpha',
        'mean_pos_mse_500step', 'best_val_mse',
    ]
    param_labels = [
        r'$\lambda_E$', r'$\lambda_P$', r'$\lambda_L$',
        'Learning rate', 'Temporal decay',
        '500-step MSE', '10-step val MSE',
    ]

    outcome_indices = {5, 6}

    n = len(param_keys)
    values = {}
    for k in param_keys:
        values[k] = np.array([float(r.get(k, np.nan)) for r in conservation])

    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x, y = values[param_keys[i]], values[param_keys[j]]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                r_val, p_val = stats.pearsonr(x[mask], y[mask])
                corr_matrix[i, j] = r_val
                p_matrix[i, j] = p_val
            else:
                corr_matrix[i, j] = np.nan
                p_matrix[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            p_val = p_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.6 else 'black'
                is_meaningful = (i in outcome_indices or j in outcome_indices)
                is_sig = p_val < 0.05 and i != j
                weight = 'bold' if (is_sig and is_meaningful) else 'normal'
                text = f'{val:.2f}'
                if is_sig and is_meaningful:
                    text += '*'
                if i == j:
                    color = '#555555'
                elif not is_meaningful:
                    color = '#999999' if abs(val) <= 0.6 else '#cccccc'
                ax.text(j, i, text, ha='center', va='center',
                        color=color, fontsize=10, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(param_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(param_labels, fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    fig.text(0.5, 0.01,
             '* p < 0.05 for correlations involving an outcome metric (MSE).\n'
             'Grey values show input-input correlations (sampling artifacts, not causally meaningful).',
             ha='center', fontsize=8, style='italic')

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    path = outdir / 'fig_a2_correlation_heatmap.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig A2: {path.name}')
    return path


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate all figures for the PINN N-body conservation paper.')
    parser.add_argument('--data-dir', default='data', help='Directory containing input data files.')
    parser.add_argument('--outdir', default='figures', help='Directory to save figure PNGs.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs_path = data_dir / 'paper_runs_final.json'
    rollout_csv = data_dir / 'rollout_length_comparison.csv'
    drift_data = data_dir / 'appendix_drift_data.json'

    runs, groups = load_runs(runs_path)

    print(f'Loaded {len(runs)} runs: ' + ', '.join(f'{k}={len(v)}' for k, v in sorted(groups.items())))
    print()

    print('Main figures:')
    fig1_mse_comparison(groups, outdir)
    fig2_conservation_drift(groups, outdir)
    fig3_parallel_coordinates(groups, outdir)
    fig4_500step_comparison(groups, outdir)
    fig5_error_growth(groups, outdir, rollout_csv)
    fig6_ablation(groups, outdir)

    # Appendix figures: keep same font sizes, just disable grid
    print('\nAppendix figures:')
    plt.rcParams.update({
        'axes.grid': False,
    })
    fig_a1_drift(outdir, drift_data)
    fig_a2_correlation_heatmap(groups, outdir)

    print('\nAll 8 figures generated.')

"""
Results Visualisation — Figure 3 Style Polar Rose Chart
=========================================================
Reproduces the polar bar (rose) diagram from Bae et al. Figure 3,
showing AUC across multiple datasets and models for LR and DM prediction.

The chart layout:
  - Each "spoke" = one dataset
  - Each bar = one model's AUC on that dataset
  - Dashed red outline = best performance on each dataset
  - Left panel  = Imaging Only
  - Right panel = Imaging + Clinical
  - Top row     = Locoregional Recurrence (LR)
  - Bottom row  = Distant Metastasis (DM)

Usage:
    # Plot published paper results (Figure 3 reproduction)
    python visualize_results.py --mode paper

    # Plot YOUR model's results after training
    python visualize_results.py --mode yours --results_dir outputs/

    # Plot both side by side
    python visualize_results.py --mode compare
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import argparse
import json
from pathlib import Path

import config


# ─── Colour palette (matches paper Figure 3) ─────────────────────────────────
MODEL_COLOURS = {
    'Clinical Baseline'    : '#1a1a1a',   # black
    'Traditional Radiomics': '#7b6fa0',   # purple
    'Mateus et al. CNN'    : '#c77b8a',   # pink-red
    'Diamant et al. CNN'   : '#b5374a',   # dark red
    'RadGraph'             : '#e8a87c',   # peach/salmon (paper colour)
    'Your Model'           : '#2196F3',   # blue (for your results)
}

DATASET_LABELS = ['HNPET', 'RADCURE', 'HN1', 'MDACC']

# ─── Published paper AUCs (Figure 3, read from paper) ────────────────────────
# Format: {model: [HNPET, RADCURE, HN1, MDACC]}

PAPER_AUCS = {
    'LR': {
        'imaging_only': {
            'Clinical Baseline'    : [0.587, 0.653, 0.664, 0.574],
            'Traditional Radiomics': [0.544, 0.641, 0.581, 0.467],
            'Mateus et al. CNN'    : [0.472, 0.389, 0.636, 0.615],
            'Diamant et al. CNN'   : [0.571, 0.731, 0.655, 0.574],
            'RadGraph'             : [0.682, 0.441, 0.473, 0.643],
        },
        'imaging_and_clinical': {
            'Clinical Baseline'    : [0.587, 0.682, 0.664, 0.574],
            'Traditional Radiomics': [0.646, 0.441, 0.350, 0.548],
            'Mateus et al. CNN'    : [0.502, 0.648, 0.450, 0.613],
            'Diamant et al. CNN'   : [0.738, 0.731, 0.806, 0.453],
            'RadGraph'             : [0.754, 0.834, 0.743, 0.603],
        },
    },
    'DM': {
        'imaging_only': {
            'Clinical Baseline'    : [0.604, 0.728, 0.832, 0.336],
            'Traditional Radiomics': [0.566, 0.717, 0.715, 0.359],
            'Mateus et al. CNN'    : [0.448, 0.632, 0.563, 0.427],
            'Diamant et al. CNN'   : [0.719, 0.385, 0.688, 0.621],
            'RadGraph'             : [0.587, 0.713, 0.715, 0.506],
        },
        'imaging_and_clinical': {
            'Clinical Baseline'    : [0.606, 0.587, 0.832, 0.505],
            'Traditional Radiomics': [0.770, 0.795, 0.641, 0.650],
            'Mateus et al. CNN'    : [0.418, 0.704, 0.567, 0.426],
            'Diamant et al. CNN'   : [0.790, 0.380, 0.689, 0.248],
            'RadGraph'             : [0.770, 0.786, 0.867, 0.559],
        },
    },
}


# ─── Core plotting function ───────────────────────────────────────────────────

def polar_rose_chart(
    ax,
    auc_dict,
    dataset_labels,
    title='',
    highlight_best=True,
):
    """
    Draw a polar bar (rose) chart on an existing Axes object.

    Parameters
    ----------
    ax             : matplotlib Axes  (must be polar)
    auc_dict       : dict  {model_name: [auc_per_dataset]}
    dataset_labels : list[str]
    title          : str
    highlight_best : bool  — outline best bar per dataset with dashed red
    """
    n_datasets = len(dataset_labels)
    n_models   = len(auc_dict)

    # Angular positions: one spoke per dataset
    angles      = np.linspace(0, 2 * np.pi, n_datasets, endpoint=False)
    bar_width   = (2 * np.pi / n_datasets) / (n_models + 1)

    model_names = list(auc_dict.keys())

    # Find best model per dataset for dashed outline
    best_per_dataset = {}
    if highlight_best:
        for ds_idx in range(n_datasets):
            best_auc   = -1
            best_model = None
            for model in model_names:
                auc = auc_dict[model][ds_idx]
                if auc > best_auc:
                    best_auc   = auc
                    best_model = model
            best_per_dataset[ds_idx] = (best_model, best_auc)

    for m_idx, model in enumerate(model_names):
        colour = MODEL_COLOURS.get(model, '#888888')
        aucs   = auc_dict[model]

        for ds_idx, auc in enumerate(aucs):
            theta = angles[ds_idx] + m_idx * bar_width - (n_models * bar_width / 2)

            bar = ax.bar(
                theta, auc,
                width     = bar_width * 0.9,
                bottom    = 0,
                color     = colour,
                alpha     = 0.85,
                edgecolor = 'white',
                linewidth = 0.4,
            )

            # Dashed red outline for best performer
            if highlight_best and best_per_dataset.get(ds_idx, (None,))[0] == model:
                ax.bar(
                    theta, auc,
                    width     = bar_width * 0.9,
                    bottom    = 0,
                    color     = 'none',
                    edgecolor = '#cc0000',
                    linewidth = 1.8,
                    linestyle = '--',
                )

            # AUC label on bar
            if auc > 0.3:
                label_r = auc + 0.02
                ax.text(
                    theta, label_r,
                    f'{auc:.3f}',
                    ha        = 'center',
                    va        = 'bottom',
                    fontsize  = 5.5,
                    rotation  = np.degrees(theta),
                    rotation_mode='anchor',
                )

    # Dataset spoke labels
    ax.set_xticks(angles)
    ax.set_xticklabels(dataset_labels, fontsize=10, fontweight='bold')

    # Radial axis
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_rlabel_position(90)

    # Grid styling
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.spines['polar'].set_visible(False)

    if title:
        ax.set_title(title, pad=18, fontsize=12, fontweight='bold')


# ─── Figure 3 reproduction ────────────────────────────────────────────────────

def plot_figure3(auc_data=None, save_path=None, extra_model=None):
    """
    Reproduce Figure 3 from the paper as a 2×2 polar rose chart grid.

    Parameters
    ----------
    auc_data    : dict or None  — if None uses PAPER_AUCS
    save_path   : Path or None
    extra_model : dict or None  — your model's AUCs to overlay
                  Format: {'LR': {'imaging_only': [hnpet, radcure, hn1, mdacc],
                                  'imaging_and_clinical': [...]},
                           'DM': {...}}
    """
    data = auc_data or PAPER_AUCS

    fig = plt.figure(figsize=(18, 18))
    fig.patch.set_facecolor('white')

    # Row labels
    row_tasks  = ['LR', 'DM']
    col_modes  = ['imaging_only', 'imaging_and_clinical']
    col_titles = ['Imaging Only', 'Imaging and Clinical']
    row_titles = ['Locoregional Recurrence', 'Distant Metastasis']

    for row_idx, task in enumerate(row_tasks):
        for col_idx, mode in enumerate(col_modes):
            ax = fig.add_subplot(
                2, 2, row_idx * 2 + col_idx + 1,
                projection='polar'
            )

            auc_dict = dict(data[task][mode])   # copy

            # Overlay your model if provided
            if extra_model and task in extra_model and mode in extra_model[task]:
                auc_dict['Your Model'] = extra_model[task][mode]

            polar_rose_chart(
                ax            = ax,
                auc_dict      = auc_dict,
                dataset_labels= DATASET_LABELS,
                title         = col_titles[col_idx] if row_idx == 0 else '',
                highlight_best= True,
            )

            # Row title on the left column
            if col_idx == 0:
                ax.text(
                    -0.15, 0.5, row_titles[row_idx],
                    transform  = ax.transAxes,
                    fontsize   = 13,
                    fontweight = 'bold',
                    va         = 'center',
                    ha         = 'right',
                    rotation   = 90,
                )

    # Legend
    legend_models = list(MODEL_COLOURS.keys())
    if extra_model:
        legend_models = legend_models  # 'Your Model' already included
    patches = [
        mpatches.Patch(color=MODEL_COLOURS[m], label=m)
        for m in legend_models
        if m in MODEL_COLOURS
    ]
    # Add dashed red outline entry
    patches.append(
        mpatches.Patch(
            facecolor='none',
            edgecolor='#cc0000',
            linestyle='--',
            linewidth=2,
            label='Best per dataset'
        )
    )

    fig.legend(
        handles  = patches,
        loc      = 'lower center',
        ncol     = len(patches),
        fontsize = 10,
        frameon  = False,
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.suptitle(
        'Model Performance on Outcome Prediction Tasks\n'
        '(AUC across datasets — RadGraph vs Baselines)',
        fontsize=15, fontweight='bold', y=1.01
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI,
                    bbox_inches='tight', facecolor='white')
        print(f"Figure 3 saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ─── Load your model results ──────────────────────────────────────────────────

def load_your_results(results_dir):
    """
    Load AUC results from your trained model output files.

    Expects: outputs/metrics_LR.json and outputs/metrics_DM.json

    Returns
    -------
    extra_model : dict  formatted for plot_figure3(extra_model=...)
                  or None if files not found
    """
    results_dir = Path(results_dir)
    extra_model = {}

    for task in ('LR', 'DM'):
        metrics_file = results_dir / f'metrics_{task}.json'
        if not metrics_file.exists():
            print(f"  Not found: {metrics_file}")
            continue

        with open(metrics_file) as f:
            m = json.load(f)

        auc = m.get('auc', 0.0)

        # Your single dataset AUC goes in all 4 dataset slots
        # (Replace with actual per-dataset values when you have them)
        extra_model[task] = {
            'imaging_only'         : [auc, auc, auc, auc],
            'imaging_and_clinical' : [auc, auc, auc, auc],
        }
        print(f"  Loaded {task} AUC: {auc:.4f}")

    return extra_model if extra_model else None


# ─── Single dataset bar chart ─────────────────────────────────────────────────

def plot_single_dataset_comparison(
    your_auc, baseline_auc, task='LR', dataset_name='CMC Vellore', save_path=None
):
    """
    Simple horizontal bar chart comparing your model vs baseline
    for your single CMC Vellore dataset.

    Parameters
    ----------
    your_auc      : float
    baseline_auc  : float
    task          : 'LR' or 'DM'
    dataset_name  : str
    save_path     : Path or None
    """
    models = ['Clinical Baseline', 'RadGraph GAT\n(Your Model)']
    aucs   = [baseline_auc, your_auc]
    colors = ['#1a1a1a', '#e8a87c']

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.barh(models, aucs, color=colors, height=0.5, edgecolor='white')

    # Value labels
    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{auc:.3f}',
            va='center', ha='left', fontsize=12, fontweight='bold'
        )

    ax.set_xlim(0, 1.0)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(
        f'Model Comparison — {task} Prediction\n{dataset_name}',
        fontsize=13, fontweight='bold'
    )
    ax.axvline(0.5, color='grey', linestyle='--', lw=1, alpha=0.5, label='Random')
    ax.set_facecolor('#f8f8f8')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 3 style polar rose chart',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Modes:
  paper   — Reproduce Figure 3 from the paper (published AUCs)
  yours   — Show your model's AUC on the chart
  compare — Side by side comparison chart (single dataset)
        """
    )
    parser.add_argument('--mode',        type=str, default='paper',
                        choices=['paper', 'yours', 'compare'],
                        help='What to plot')
    parser.add_argument('--results_dir', type=str,
                        default=str(config.OUTPUT_DIR),
                        help='Directory containing metrics_LR.json etc.')
    parser.add_argument('--save_dir',    type=str,
                        default=str(config.OUTPUT_DIR),
                        help='Directory to save figures')
    parser.add_argument('--task',        type=str, default='LR',
                        choices=['LR', 'DM'],
                        help='Task for single-dataset comparison chart')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'paper':
        print("Plotting Figure 3 (published paper AUCs)...")
        plot_figure3(
            save_path = save_dir / 'figure3_paper.png'
        )

    elif args.mode == 'yours':
        print("Loading your model results...")
        extra_model = load_your_results(args.results_dir)
        if extra_model is None:
            print("No results found. Train your model first.")
            print("Run: python main.py --task LR --stage train")
            return
        plot_figure3(
            extra_model = extra_model,
            save_path   = save_dir / 'figure3_with_your_model.png'
        )

    elif args.mode == 'compare':
        # Load your AUC and baseline
        metrics_file = Path(args.results_dir) / f'metrics_{args.task}.json'
        baseline_file= Path(args.results_dir) / f'test_results_{args.task}.csv'

        if not metrics_file.exists():
            print(f"Model results not found: {metrics_file}")
            return

        with open(metrics_file) as f:
            your_auc = json.load(f).get('auc', 0.0)

        baseline_auc = 0.0
        if baseline_file.exists():
            import pandas as pd
            from sklearn.metrics import roc_auc_score
            df           = pd.read_csv(baseline_file)
            baseline_auc = roc_auc_score(
                df['true_label'], df['predicted_prob']
            )

        print(f"\nYour GAT AUC  : {your_auc:.4f}")
        print(f"Baseline AUC  : {baseline_auc:.4f}")

        plot_single_dataset_comparison(
            your_auc     = your_auc,
            baseline_auc = baseline_auc,
            task         = args.task,
            dataset_name = 'CMC Vellore',
            save_path    = save_dir / f'comparison_{args.task}.png'
        )

    print("\nDone! Check outputs/ for saved figures.")


if __name__ == '__main__':
    main()

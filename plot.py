"""
plot.py

Generate comparison plots for reconstructed vs. truth jet observables:
- Jet pT
- Jet mass
- Jet girth
- Energy correlation function (ECF)

Loads NumPy arrays from the 'obs' directory and saves figures to 'figs'.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def configure_matplotlib():
    """Set global matplotlib parameters for consistent styling."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "figure.dpi": 125
    })


def create_ratio_axes(figsize=(10, 10), height_ratios=(3, 1), hspace=0):
    """Create a figure with a main panel and a ratio subplot and return (fig, ax_main, ax_ratio)."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, hspace=hspace, height_ratios=height_ratios)
    ax_main, ax_ratio = gs.subplots(sharex=True)
    return fig, ax_main, ax_ratio


def plot_pt(obs_dir, figs_dir):
    """Plot jet pT distribution and ratio."""
    truth = np.load(obs_dir / "truth_pt.npy")
    reco = np.load(obs_dir / "reco_pt.npy")
    ics = np.load(obs_dir / "ics_pt.npy")
    bins = np.linspace(100, 1200, 20)

    fig, ax_main, ax_ratio = create_ratio_axes()
    # Main histogram
    n_reco, _, _ = ax_main.hist(
        reco, bins,
        histtype='step', linewidth=2, linestyle='dashed',
        label='\\textsc{DeepSub} (Ours)', density=True, color='red'
    )
    n_truth, _, _ = ax_main.hist(
        truth, bins, alpha=0.5,
        label='Truth', density=True
    )
    n_ics, _, _ = ax_main.hist(
        ics, bins,
        histtype='step', linewidth=2, linestyle='dashed',
        label='Event-Wide ICS', density=True, color='green'
    )
    ax_main.set_yscale('log')
    ax_main.set_ylabel('Events [a.u.]', fontsize=30)
    ax_main.legend(loc='upper right', fontsize=18)

    # Ratio subplot
    ax_ratio.set_xlabel('Jet $p_{\\mathrm{T}}$ [GeV]', fontsize=30)
    ax_ratio.set_ylabel(r'$\\frac{\\textrm{Predicted}}{\\textrm{Truth}}$', fontsize=30)
    ratio = n_reco / n_truth
    ratio_ics = n_ics / n_truth
    ax_ratio.hist(
        bins[:-1], bins, weights=ratio,
        histtype='step', linewidth=2, linestyle='dashed', color='red'
    )
    ax_ratio.hist(
        bins[:-1], bins, weights=ratio_ics,
        histtype='step', linewidth=2, linestyle='dashed', color='green'
    )
    ax_ratio.plot([0, 123232323], [1, 1], color='black', linestyle='dashed')
    ax_ratio.set_ylim([0, 2])

    # Annotation
    ax_main.text(
        560, 1.55*0.255e-3,
        "\\textsc{Jewel} Dijets, $\\sqrt{s_{\\mathrm{NN}}}=5.02$ TeV,\n"
        "0-10\\% Centrality, $v_2=0.05$, $\\widehat{p}_{\\mathrm{T}}> 100$ GeV,\n"
        "Anti-$k_{\\mathrm{T}}$, $R=0.4$, $p_{\\mathrm{T}}^{\\mathrm{jet}} > 100$ GeV.",
        fontsize=18
    )

    out_path = figs_dir / "jetPT.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_mass(obs_dir, figs_dir):
    """Plot jet mass distribution and ratio."""
    truth = np.load(obs_dir / "truth_mass.npy")
    reco = np.load(obs_dir / "reco_mass.npy")
    ics = np.load(obs_dir / "ics_mass.npy")
    bins = np.linspace(3, 140, 20)

    fig, ax_main, ax_ratio = create_ratio_axes()
    n_reco, _, _ = ax_main.hist(
        reco, bins,
        histtype='step', linewidth=2, linestyle='dashed',
        label='\\textsc{DeepSub} (Ours)', density=True, color='red'
    )
    n_truth, _, _ = ax_main.hist(
        truth, bins, alpha=0.5,
        label='Truth', density=True
    )
    n_ics, _, _ = ax_main.hist(
        ics, bins,
        histtype='step', linewidth=2, linestyle='dashed',
        label='Event-Wide ICS', density=True, color='green'
    )
    ax_main.set_yscale('log')
    ax_main.set_xlim(3, 140)
    ax_main.set_ylim(0.25e-4, 0.9e-1)
    ax_main.set_ylabel('Events [a.u.]', fontsize=30)
    ax_main.legend(loc='upper right', fontsize=18)

    ax_ratio.set_xlabel('Jet Mass [GeV]', fontsize=30)
    ax_ratio.set_ylabel(r'$\\frac{\\textrm{Predicted}}{\\textrm{Truth}}$', fontsize=30)
    ratio = n_reco / n_truth
    ratio_ics = n_ics / n_truth
    ax_ratio.hist(
        bins[:-1], bins, weights=ratio,
        histtype='step', linewidth=2, linestyle='dashed', color='red'
    )
    ax_ratio.hist(
        bins[:-1], bins, weights=ratio_ics,
        histtype='step', linewidth=2, linestyle='dashed', color='green'
    )
    ax_ratio.plot([0, 123232323], [1, 1], color='black', linestyle='dotted')
    ax_ratio.set_ylim([0, 2])

    ax_main.text(
        62, 1.75*0.225e-2,
        "\\textsc{Jewel} Dijets, $\\sqrt{s_{\\mathrm{NN}}}=5.02$ TeV,\n"
        "0-10\\% Centrality, $v_2=0.05$, $\\widehat{p}_{\\mathrm{T}}> 100$ GeV,\n"
        "Anti-$k_{\\mathrm{T}}$, $R=0.4$, $p_{\\mathrm{T}}^{\\mathrm{jet}} > 100$ GeV.",
        fontsize=18
    )

    out_path = figs_dir / "jet_mass.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_girth(obs_dir, figs_dir):
    """Plot jet girth distribution and ratio."""
    truth = np.load(obs_dir / "truth_girth.npy")
    reco = np.load(obs_dir / "reco_girth.npy")
    ics = np.load(obs_dir / "ics_girth.npy")
    bins = np.linspace(0.05, 0.25, 19)

    fig, ax_main, ax_ratio = create_ratio_axes()
    n_reco, _, _ = ax_main.hist(
        reco, bins,
        histtype='step', linewidth=2, linestyle='dashed',
        label='\\textsc{DeepSub} (Ours)', density=True, color='red'
    )
    n_truth, _, _ = ax_main.hist(
        truth, bins, alpha=0.5,
        label='Truth', density=True
    )
    n_ics, _, _ = ax_main.hist(
        ics, bins,
        histtype='step', linewidth=2, linestyle='dashed',
        label='Event-Wide ICS', density=True, color='green'
    )
    ax_main.set_yscale('log')
    ax_main.set_xlim(0.05, 0.25)
    ax_main.set_ylim(0.5, 10)
    ax_main.set_ylabel('Events [a.u.]', fontsize=30)
    ax_main.legend(loc='upper right', fontsize=18)

    ax_ratio.set_xlabel('Jet Girth', fontsize=30)
    ax_ratio.set_ylabel(r'$\\frac{\\textrm{Predicted}}{\\textrm{Truth}}$', fontsize=30)
    ratio = n_reco / n_truth
    ratio_ics = n_ics / n_truth
    ax_ratio.hist(
        bins[:-1], bins, weights=ratio,
        histtype='step', linewidth=2, linestyle='dashed', color='red'
    )
    ax_ratio.hist(
        bins[:-1], bins, weights=ratio_ics,
        histtype='step', linewidth=2, linestyle='dashed', color='green'
    )
    ax_ratio.plot([0, 123232323], [1, 1], color='black', linestyle='dashed')
    ax_ratio.set_ylim([0, 2])

    ax_main.text(
        0.075, 1.45,
        "\\textsc{Jewel} Dijets, $\\sqrt{s_{\\mathrm{NN}}}=5.02$ TeV,\n"
        "0-10\\% Centrality, $v_2=0.05$, $\\widehat{p}_{\\mathrm{T}}> 100$ GeV,\n"
        "Anti-$k_{\\mathrm{T}}$, $R=0.4$, $p_{\\mathrm{T}}^{\\mathrm{jet}} > 100$ GeV.",
        fontsize=18
    )

    out_path = figs_dir / "jet_girth.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_ecf(obs_dir, figs_dir):
    """Plot energy correlation function (ECF) distribution and ratio."""
    truth = np.load(obs_dir / "truth_ecf.npy")
    reco = np.load(obs_dir / "reco_ecf.npy")
    ics = np.load(obs_dir / "ics_ecf.npy")
    bins = np.logspace(3, 6, 21)

    fig, ax_main, ax_ratio = create_ratio_axes()
    n_reco, _, _ = ax_main.hist(
        reco, bins,
        histtype='step', linewidth=2, linestyle='dashed',
        label='\\textsc{DeepSub} (Ours)', density=True, color='red'
    )
    n_truth, _, _ = ax_main.hist(
        truth, bins, alpha=0.5,
        label='Truth', density=True
    )
    n_ics, _, _ = ax_main.hist(
        ics, bins,
        histtype='step', linewidth=2, linestyle='dashed',
        label='Event-Wide ICS', density=True, color='green'
    )
    ax_main.set_yscale('log')
    ax_main.set_xscale('log')
    ax_main.set_xlim(3e3, 1e6)
    ax_main.set_ylabel('Events [a.u.]', fontsize=30)
    ax_main.legend(loc='upper right', fontsize=18)

    ax_ratio.set_xlabel(r'$\\mathrm{ECF}_{N=2}^{\\beta=2}$', fontsize=30)
    ax_ratio.set_ylabel(r'$\\frac{\\textrm{Predicted}}{\\textrm{Truth}}$', fontsize=30)
    ratio = n_reco / n_truth
    ratio_ics = n_ics / n_truth
    ax_ratio.hist(
        bins[:-1], bins, weights=ratio,
        histtype='step', linewidth=2, linestyle='dashed', color='red'
    )
    ax_ratio.hist(
        bins[:-1], bins, weights=ratio_ics,
        histtype='step', linewidth=2, linestyle='dashed', color='green'
    )
    ax_ratio.plot([0, 123232323], [1, 1], color='black', linestyle='dotted')
    ax_ratio.set_ylim([0, 2])
    ax_ratio.set_xscale('log')

    ax_main.text(
        5000, 0.325e-8,
        "\\textsc{Jewel} Dijets, $\\sqrt{s_{\\mathrm{NN}}}=5.02$ TeV,\n"
        "0-10\\% Centrality, $v_2=0.05$, $\\widehat{p}_{\\mathrm{T}}> 100$ GeV,\n"
        "Anti-$k_{\\mathrm{T}}$, $R=0.4$, $p_{\\mathrm{T}}^{\\mathrm{jet}} > 100$ GeV.",
        fontsize=18
    )

    out_path = figs_dir / "ecf.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=400)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    """Main entry point: configure plotting and generate all figures."""
    obs_dir = Path("obs")
    figs_dir = Path("figs")
    figs_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    plot_pt(obs_dir, figs_dir)
    plot_mass(obs_dir, figs_dir)
    plot_girth(obs_dir, figs_dir)
    plot_ecf(obs_dir, figs_dir)


if __name__ == "__main__":
    main()


"""
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
import mplhep as hep


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


def _hist_stats(data: np.ndarray, bins: np.ndarray):
    """Return (counts, densities, poisson_sigma)."""
    counts, _ = np.histogram(data, bins)
    widths = np.diff(bins)
    total = counts.sum() if counts.sum() > 0 else 1  # avoid div-by-zero
    dens = counts / (total * widths)
    sigma = np.sqrt(counts) / (total * widths)
    return counts, dens, sigma


def _ratio(numer_dens, numer_sigma, denom_dens, denom_sigma):
    """Return ratio and its propagated uncertainty."""
    ratio = np.divide(numer_dens, denom_dens, out=np.zeros_like(numer_dens), where=denom_dens != 0)
    sigma = np.zeros_like(ratio)
    mask = (numer_dens > 0) & (denom_dens > 0)
    sigma[mask] = ratio[mask] * np.sqrt((numer_sigma[mask] / numer_dens[mask]) ** 2 + (denom_sigma[mask] / denom_dens[mask]) ** 2)
    return ratio, sigma


def _annotate(ax, text, **kwargs):
    if text:
        ax.text(*text, fontsize=19, transform=ax.transData, **kwargs)


def hist_ratio_plot(truth, reco, ics, *, bins, xlabel, out_name, xscale="linear", xlim=None, annotation=None):
    """Generic histogram + ratio plot with mplhep styling."""

    # Statistics
    cnt_reco, dens_reco, sig_reco = _hist_stats(reco, bins)
    cnt_ics,  dens_ics,  sig_ics  = _hist_stats(ics,  bins)
    cnt_truth, dens_truth, sig_truth = _hist_stats(truth, bins)

    ratio_reco, sig_ratio_reco = _ratio(dens_reco, sig_reco, dens_truth, sig_truth)
    ratio_ics,  sig_ratio_ics  = _ratio(dens_ics,  sig_ics,  dens_truth, sig_truth)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, height_ratios=[3, 1], hspace=0)
    ax_main, ax_ratio = gs.subplots(sharex=True)

    hep.histplot(cnt_reco, bins, yerr=True, ax=ax_main, density=True,
                 histtype="step", color="red",   linestyle="dashed", linewidth=2,
                 label="\\textsc{DeepSub} (Ours)")
    hep.histplot(cnt_ics,  bins, yerr=True, ax=ax_main, density=True,
                 histtype="step", color="green", linestyle="dashed", linewidth=2,
                 label="Event-Wide ICS")
    hep.histplot(cnt_truth, bins, yerr=True, ax=ax_main, density=True,
                 histtype="fill", alpha=0.5, label="Truth")
    hep.histplot(cnt_truth, bins, yerr=True, ax=ax_main, density=True,
                 histtype="step", alpha=0.5, color="#1f77b4")

    ax_main.set_ylabel("Events [a.u.]", fontsize=30)
    ax_main.set_yscale("log")
    ax_main.legend(fontsize=19, loc="upper right")

    _annotate(ax_main, annotation)

    # Ratio pad and truth uncertainty band
    lower = 1 - sig_truth / dens_truth
    upper = 1 + sig_truth / dens_truth
    lower = np.append(lower, lower[-1])
    upper = np.append(upper, upper[-1])
    ax_ratio.fill_between(bins, lower, upper, step="post", color="yellow", alpha=0.5, label="Truth uncertainty")

    hep.histplot(ratio_reco, bins, yerr=sig_ratio_reco, ax=ax_ratio,
                 histtype="step", color="red", linestyle="dashed", linewidth=2,
                 label="DeepSub / Truth")
    hep.histplot(ratio_ics,  bins, yerr=sig_ratio_ics,  ax=ax_ratio,
                 histtype="step", color="green", linestyle="dashed", linewidth=2,
                 label="ICS / Truth")

    ax_ratio.axhline(1, linestyle="dashed", color="black")
    ax_ratio.set_xlabel(xlabel, fontsize=30)
    ax_ratio.set_ylabel(r"$\\frac{\\textrm{Predicted}}{\\textrm{Truth}}$", fontsize=30)
    ax_ratio.set_ylim(0, 2)

    # Common cosmetics
    if xscale == "log":
        ax_main.set_xscale("log")
        ax_ratio.set_xscale("log")
    if xlim:
        ax_main.set_xlim(*xlim)

    plt.savefig(out_name, bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_pt(obs_dir: Path, figs_dir: Path):
    truth = np.load(obs_dir / "truth_pt.npy")
    reco  = np.load(obs_dir / "reco_pt.npy")
    ics   = np.load(obs_dir / "ics_pt.npy")
    bins = np.linspace(100, 1200, 20)
    annotation = (180, 1.55e-5, "\\textsc{Jewel} Dijets, $\\sqrt{s_{\\mathrm{NN}}}=5.02$ TeV,\n0-10\\% Centrality, $v_2=0.05$, $\\widehat{p}_{\\mathrm{T}}> 100$ GeV,\n Anti-$k_{\\mathrm{T}}$, $R=0.4$, $p_{\\mathrm{T}}^{\\mathrm{jet}} > 100$ GeV.")
    hist_ratio_plot(truth, reco, ics, bins=bins, xlabel='Jet $p_{\\mathrm{T}}$ [GeV]',
                    out_name=figs_dir / "jetPT.png", xlim=(100, 1200), annotation=annotation)


def plot_mass(obs_dir: Path, figs_dir: Path):
    truth = np.load(obs_dir / "truth_mass.npy")
    reco  = np.load(obs_dir / "reco_mass.npy")
    ics   = np.load(obs_dir / "ics_mass.npy")
    bins = np.linspace(3, 140, 20)
    annotation = (12, 0.7e-4, "\\textsc{Jewel} Dijets, $\\sqrt{s_{\\mathrm{NN}}}=5.02$ TeV,\n 0-10\\% Centrality, $v_2=0.05$, $\\widehat{p}_{\\mathrm{T}}> 100$ GeV,\n Anti-$k_{\\mathrm{T}}$, $R=0.4$, $p_{\\mathrm{T}}^{\\mathrm{jet}} > 100$ GeV.")
    hist_ratio_plot(truth, reco, ics, bins=bins, xlabel='Jet Mass [GeV]',
                    out_name=figs_dir / "jet_mass.png", xlim=(3, 140), annotation=annotation)


def plot_girth(obs_dir: Path, figs_dir: Path):
    truth = np.load(obs_dir / "truth_girth.npy")
    reco  = np.load(obs_dir / "reco_girth.npy")
    ics   = np.load(obs_dir / "ics_girth.npy")
    bins = np.linspace(0.05, 0.25, 19)
    annotation = (0.075, 0.65, "\\textsc{Jewel} Dijets, $\\sqrt{s_{\\mathrm{NN}}}=5.02$ TeV,\n0-10\\% Centrality, $v_2=0.05$, $\\widehat{p}_{\\mathrm{T}}> 100$ GeV,\n Anti-$k_{\\mathrm{T}}$, $R=0.4$, $p_{\\mathrm{T}}^{\\mathrm{jet}} > 100$ GeV.")
    hist_ratio_plot(truth, reco, ics, bins=bins, xlabel='Jet Girth',
                    out_name=figs_dir / "jet_girth.png", xlim=(0.05, 0.25), annotation=annotation)


def plot_ecf(obs_dir: Path, figs_dir: Path):
    truth = np.load(obs_dir / "truth_ecf.npy")
    reco  = np.load(obs_dir / "reco_ecf.npy")
    ics   = np.load(obs_dir / "ics_ecf.npy")
    bins = np.logspace(3, 6, 21)
    annotation = (2.3 * 0.25e4, 0.225e-7, "\\textsc{Jewel} Dijets, $\\sqrt{s_{\\mathrm{NN}}}=5.02$ TeV,\n0-10\\% Centrality, $v_2=0.05$, $\\widehat{p}_{\\mathrm{T}}> 100$ GeV,\n Anti-$k_{\\mathrm{T}}$, $R=0.4$, $p_{\\mathrm{T}}^{\\mathrm{jet}} > 100$ GeV.")
    hist_ratio_plot(truth, reco, ics, bins=bins, xlabel=r'$\\mathrm{ECF}_{N=2}^{\\beta=2}$',
                    out_name=figs_dir / "ecf.png", xscale="log", xlim=(3e3, 1e6), annotation=annotation)


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

"""
plot_sobol_indices.py
=====================
Reads a UQEF-Dynamic simulation output directory and prints / plots all
computed Sobol sensitivity indices:

  • Time-wise first-order  (Sobol_m)   – line plot over time per QoI
  • Time-wise total-order  (Sobol_t)   – line plot over time per QoI
  • Generalized Sobol indices           – bar chart per QoI (scalar summary)

Data is read directly from the saved pickle files; no UQEF-Dynamic import
is required.

Usage
-----
    python plot_sobol_indices.py <outputDir> [options]

    python plot_sobol_indices.py debug_output/battery_mc_10000_kl_trunc_local_debug
    python plot_sobol_indices.py debug_output/battery_mc_70000_kl_local_debug \\
        --qoi V_sol --clip-negative --save-dir figures/

Options
-------
    --qoi QOI           Only process the named QoI (default: all found)
    --no-timewise       Skip time-wise Sobol line plots
    --no-generalized    Skip generalized Sobol bar charts
    --clip-negative     Clip negative Sobol values to 0 before plotting
    --top N             Only show the top-N most influential parameters
                        (ranked by mean |Sobol_m| or generalized index)
    --save-dir DIR      Save figures to DIR instead of showing interactively
    --dpi DPI           Figure DPI for saved files (default: 150)
    --no-latex          Disable matplotlib LaTeX rendering
    --verbose, -v       Print extra diagnostics
"""

import argparse
import pathlib
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirror utility.py without importing the whole framework)
# ─────────────────────────────────────────────────────────────────────────────

TIME_COL = "TimeStamp"
SOBOL_M   = "Sobol_m"
SOBOL_T   = "Sobol_t"
GEN_TOTAL = "generalized_sobol_total_index_"
GEN_MAIN  = "generalized_sobol_main_index_"


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_pickle(path: pathlib.Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_df_pickle(path: pathlib.Path) -> pd.DataFrame | None:
    """Try gzip-compressed first, then plain pickle."""
    for compression in ("gzip", None, "infer"):
        try:
            return pd.read_pickle(str(path), compression=compression)
        except Exception:
            pass
    return None


def discover_qois(workingDir: pathlib.Path) -> list[str]:
    """Return QoI names found from statistics_dictionary_qoi_<QOI>.pkl files."""
    pattern = "statistics_dictionary_qoi_*.pkl"
    qois = []
    for p in sorted(workingDir.glob(pattern)):
        stem = p.stem  # e.g. "statistics_dictionary_qoi_V_sol"
        qoi = stem[len("statistics_dictionary_qoi_"):]
        qois.append(qoi)
    return qois


def load_stats_dict(workingDir: pathlib.Path, qoi: str) -> dict | None:
    """Load statistics_dictionary_qoi_<qoi>.pkl → {timestamp: {key: value}}."""
    path = workingDir / f"statistics_dictionary_qoi_{qoi}.pkl"
    if not path.exists():
        return None
    try:
        return _load_pickle(path)
    except Exception as e:
        print(f"  [WARN] Could not load {path.name}: {e}")
        return None


_NON_PARAM_COLS = {"Index_run", "successful_run", "GOF", "TimeStamp", "qoi",
                   "measured", "measured_norm"}


def get_param_names_from_df(workingDir: pathlib.Path) -> list[str]:
    """Extract uncertain parameter names from df_index_parameter.pkl columns."""
    for fname in ("df_index_parameter.pkl",):
        p = workingDir / fname
        if p.exists():
            df = _load_df_pickle(p)
            if df is not None:
                cols = [c for c in df.columns if c not in _NON_PARAM_COLS]
                if cols:
                    return cols
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_timewise_sobol(
    stats: dict,
    param_names: list[str],
    si_key: str,
    clip_negative: bool,
) -> pd.DataFrame | None:
    """
    Extract time-wise Sobol indices (Sobol_m or Sobol_t) from stats dict.

    Returns a DataFrame with columns [TIME_COL, param_0, param_1, ...] or None
    if the key is not present.
    """
    timestamps = sorted(stats.keys())
    if not timestamps:
        return None

    # Check presence
    ts0 = timestamps[0]
    if si_key not in stats[ts0]:
        return None

    rows = []
    for ts in timestamps:
        entry = stats[ts].get(si_key)
        if entry is None:
            continue
        arr = np.asarray(entry, dtype=float)
        rows.append([ts] + arr.tolist())

    if not rows:
        return None

    n_params = len(rows[0]) - 1
    if param_names and len(param_names) == n_params:
        col_names = [TIME_COL] + list(param_names)
    else:
        col_names = [TIME_COL] + [f"param_{i}" for i in range(n_params)]

    df = pd.DataFrame(rows, columns=col_names)
    if clip_negative:
        for c in col_names[1:]:
            df[c] = df[c].clip(lower=0.0)
    return df


def extract_generalized_sobol(
    stats: dict,
    prefix: str,
) -> dict[str, float] | None:
    """
    Extract scalar generalized Sobol indices.

    Searches all timestamps for keys starting with `prefix`.
    If a key appears at multiple timestamps (over-time variant), returns the
    last non-NaN value; otherwise returns the single scalar.
    Returns dict {param_name: value} or None.
    """
    timestamps = sorted(stats.keys())
    result: dict[str, list] = {}

    for ts in timestamps:
        for key, val in stats[ts].items():
            if isinstance(key, str) and key.startswith(prefix):
                param = key[len(prefix):]
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                result.setdefault(param, []).append(fval)

    if not result:
        return None

    # Use the last valid value per parameter
    return {p: vals[-1] for p, vals in result.items()}


def extract_generalized_sobol_over_time(
    stats: dict,
    prefix: str,
) -> pd.DataFrame | None:
    """
    If generalized Sobol indices were computed at every timestep
    (compute_generalized_sobol_indices_over_time=True), return a DataFrame
    with columns [TIME_COL, param_0, param_1, ...].
    """
    timestamps = sorted(stats.keys())
    # Collect all param names from the last timestamp
    last_ts = timestamps[-1]
    param_names = [
        k[len(prefix):]
        for k in stats[last_ts]
        if isinstance(k, str) and k.startswith(prefix)
    ]
    if not param_names:
        return None

    # Check if it's actually time-varying (more than 1 timestamp has values)
    ts_with_gen = [
        ts for ts in timestamps
        if any(isinstance(k, str) and k.startswith(prefix)
               for k in stats[ts])
    ]
    if len(ts_with_gen) <= 1:
        return None  # only at final ts → not over-time

    rows = []
    for ts in ts_with_gen:
        row = [ts]
        for p in param_names:
            key = prefix + p
            try:
                row.append(float(stats[ts].get(key, np.nan)))
            except (TypeError, ValueError):
                row.append(np.nan)
        rows.append(row)

    return pd.DataFrame(rows, columns=[TIME_COL] + param_names)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking / top-N selection
# ─────────────────────────────────────────────────────────────────────────────

def top_n_params(df: pd.DataFrame, n: int | None) -> list[str]:
    """Return parameter columns sorted by descending mean absolute value."""
    param_cols = [c for c in df.columns if c != TIME_COL]
    if not param_cols:
        return []
    means = df[param_cols].abs().mean().sort_values(ascending=False)
    if n is not None and n < len(means):
        return means.index[:n].tolist()
    return means.index.tolist()


def top_n_from_dict(d: dict, n: int | None) -> dict:
    if n is None or n >= len(d):
        return d
    sorted_items = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return dict(sorted_items[:n])


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def print_timewise_summary(df: pd.DataFrame, si_label: str, qoi: str):
    """Print mean, min, max of each Sobol index over time."""
    _section(f"{si_label} over time — QoI: {qoi}")
    param_cols = [c for c in df.columns if c != TIME_COL]
    header = f"  {'Parameter':<42}  {'Mean':>10}  {'Min':>10}  {'Max':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in param_cols:
        mean_ = df[c].mean()
        min_  = df[c].min()
        max_  = df[c].max()
        print(f"  {c:<42}  {mean_:>10.5f}  {min_:>10.5f}  {max_:>10.5f}")


def print_generalized_summary(d: dict, label: str, qoi: str):
    """Print generalized Sobol index scalars sorted by descending value."""
    _section(f"{label} — QoI: {qoi}")
    print(f"  {'Parameter':<42}  {'Index':>10}")
    print("  " + "-" * 56)
    for param, val in sorted(d.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {param:<42}  {val:>10.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette that works for many parameters
_PALETTE = [
    "#0072B2", "#E69F00", "#CC79A7", "#009E73", "#56B4E9",
    "#D55E00", "#F0E442", "#000000", "#8B4513", "#6A0DAD",
    "#00CED1", "#FF69B4", "#32CD32", "#FF4500", "#1E90FF",
    "#FFD700", "#8FBC8F", "#DC143C", "#4169E1", "#20B2AA",
    "#FF1493", "#228B22", "#B8860B", "#9932CC", "#CD5C5C",
]


def _param_colors(params: list[str]) -> dict[str, str]:
    return {p: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(params)}


def plot_timewise_sobol(
    df_m: pd.DataFrame | None,
    df_t: pd.DataFrame | None,
    qoi: str,
    top_n: int | None,
    save_path: pathlib.Path | None,
    dpi: int,
):
    """Line plot of time-wise Sobol indices for a single QoI."""
    panels = []
    if df_m is not None:
        panels.append((df_m, "First-order (Sobol_m)"))
    if df_t is not None:
        panels.append((df_t, "Total-order (Sobol_t)"))
    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels),
                             sharex=True, squeeze=False)
    fig.suptitle(f"Sobol Sensitivity Indices over Time — QoI: {qoi}",
                 fontsize=13, fontweight="bold")

    for ax, (df, title) in zip(axes[:, 0], panels):
        sel_params = top_n_params(df, top_n)
        colors = _param_colors(sel_params)
        time = df[TIME_COL].values
        for p in sel_params:
            ax.plot(time, df[p].values, label=p, color=colors[p], linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Sensitivity index")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.legend(loc="upper right", fontsize=8, ncol=2,
                  framealpha=0.8, edgecolor="gray")
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel(TIME_COL)
    plt.tight_layout()

    if save_path:
        fname = save_path / f"sobol_timewise_{qoi}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {fname}")
        plt.close(fig)
    else:
        plt.show()


def plot_generalized_sobol_bar(
    gen_total: dict | None,
    gen_main: dict | None,
    qoi: str,
    top_n: int | None,
    save_path: pathlib.Path | None,
    dpi: int,
):
    """Grouped bar chart of generalized Sobol indices for a single QoI."""
    if gen_total is None and gen_main is None:
        return

    # Merge param lists
    all_params = []
    if gen_total:
        all_params += list(gen_total.keys())
    if gen_main:
        for p in gen_main:
            if p not in all_params:
                all_params.append(p)

    # Rank by total index (or main if no total)
    rank_dict = gen_total if gen_total else gen_main
    rank_dict = top_n_from_dict(rank_dict, top_n)
    params = list(rank_dict.keys())  # ordered by magnitude

    # Get values for selected params
    tot_vals  = [gen_total.get(p, np.nan)  if gen_total  else np.nan for p in params]
    main_vals = [gen_main.get(p,  np.nan)  if gen_main   else np.nan for p in params]

    n = len(params)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 0.55 + 2), 5))
    fig.suptitle(f"Generalized Sobol Indices — QoI: {qoi}",
                 fontsize=13, fontweight="bold")

    has_tot  = gen_total  is not None
    has_main = gen_main   is not None

    if has_tot and has_main:
        bars1 = ax.bar(x - width / 2, tot_vals,  width, label="Total-order",       color="#0072B2", alpha=0.85)
        bars2 = ax.bar(x + width / 2, main_vals, width, label="First-order (main)", color="#E69F00", alpha=0.85)
    elif has_tot:
        ax.bar(x, tot_vals,  width * 1.5, label="Total-order", color="#0072B2", alpha=0.85)
    else:
        ax.bar(x, main_vals, width * 1.5, label="First-order (main)", color="#E69F00", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Sensitivity index")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fname = save_path / f"sobol_generalized_{qoi}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {fname}")
        plt.close(fig)
    else:
        plt.show()


def plot_generalized_sobol_over_time(
    df_tot: pd.DataFrame | None,
    df_main: pd.DataFrame | None,
    qoi: str,
    top_n: int | None,
    save_path: pathlib.Path | None,
    dpi: int,
):
    """Line plots of over-time generalized Sobol indices (if computed)."""
    panels = []
    if df_tot  is not None:
        panels.append((df_tot,  "Generalized Total-order"))
    if df_main is not None:
        panels.append((df_main, "Generalized First-order (main)"))
    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels),
                             sharex=True, squeeze=False)
    fig.suptitle(f"Generalized Sobol Indices over Time — QoI: {qoi}",
                 fontsize=13, fontweight="bold")

    for ax, (df, title) in zip(axes[:, 0], panels):
        sel_params = top_n_params(df, top_n)
        colors = _param_colors(sel_params)
        time = df[TIME_COL].values
        for p in sel_params:
            ax.plot(time, df[p].values, label=p, color=colors[p], linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Sensitivity index")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.legend(loc="upper right", fontsize=8, ncol=2,
                  framealpha=0.8, edgecolor="gray")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel(TIME_COL)
    plt.tight_layout()

    if save_path:
        fname = save_path / f"sobol_generalized_overtime_{qoi}.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {fname}")
        plt.close(fig)
    else:
        plt.show()


def plot_heatmap_generalized(
    gen_total_by_qoi: dict[str, dict],
    gen_main_by_qoi: dict[str, dict],
    top_n: int | None,
    save_path: pathlib.Path | None,
    dpi: int,
):
    """
    Heatmap of generalized Sobol indices: rows = parameters, cols = QoIs.
    Only rendered when there is more than one QoI.
    """
    qois_tot  = [q for q, d in gen_total_by_qoi.items() if d]
    qois_main = [q for q, d in gen_main_by_qoi.items()  if d]

    for qois, data_by_qoi, label, fname_tag in [
        (qois_tot,  gen_total_by_qoi,  "Total-order",         "total"),
        (qois_main, gen_main_by_qoi,   "First-order (main)",  "main"),
    ]:
        if len(qois) < 2:
            continue

        # Union of all params (ranked by first QoI's values)
        all_params_set = {}
        for q in qois:
            for p, v in data_by_qoi[q].items():
                all_params_set[p] = all_params_set.get(p, 0) + abs(v)
        params = sorted(all_params_set, key=lambda p: all_params_set[p], reverse=True)
        if top_n:
            params = params[:top_n]

        matrix = np.full((len(params), len(qois)), np.nan)
        for j, q in enumerate(qois):
            for i, p in enumerate(params):
                matrix[i, j] = data_by_qoi[q].get(p, np.nan)

        fig, ax = plt.subplots(figsize=(max(6, len(qois) * 1.5),
                                        max(5, len(params) * 0.35 + 2)))
        fig.suptitle(f"Generalized Sobol Indices ({label}) — All QoIs",
                     fontsize=13, fontweight="bold")
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=np.nanmax(matrix))
        plt.colorbar(im, ax=ax, label="Sensitivity index")
        ax.set_xticks(range(len(qois)))
        ax.set_xticklabels(qois, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params, fontsize=8)
        ax.set_xlabel("QoI")
        ax.set_ylabel("Parameter")

        for i in range(len(params)):
            for j in range(len(qois)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if val < 0.5 * np.nanmax(matrix) else "white")
        plt.tight_layout()

        if save_path:
            fname = save_path / f"sobol_heatmap_{fname_tag}.png"
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            print(f"  Saved: {fname}")
            plt.close(fig)
        else:
            plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Print and plot Sobol sensitivity indices from a UQEF-Dynamic output directory.")
    parser.add_argument("outputDir", type=pathlib.Path,
                        help="Simulation output directory (workingDir or outputResultDir)")
    parser.add_argument("--qoi", type=str, default=None,
                        help="Only process this QoI (default: all found)")
    parser.add_argument("--no-timewise", action="store_true",
                        help="Skip time-wise Sobol line plots")
    parser.add_argument("--no-generalized", action="store_true",
                        help="Skip generalized Sobol bar charts")
    parser.add_argument("--clip-negative", action="store_true",
                        help="Clip negative Sobol index values to 0")
    parser.add_argument("--top", type=int, default=None, metavar="N",
                        help="Only show top-N most influential parameters")
    parser.add_argument("--save-dir", type=pathlib.Path, default=None,
                        help="Directory to save figures (default: show interactively)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved figures (default: 150)")
    parser.add_argument("--no-latex", action="store_true",
                        help="Disable matplotlib LaTeX rendering")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    outputDir = args.outputDir.resolve()
    if not outputDir.exists():
        print(f"ERROR: directory does not exist: {outputDir}")
        sys.exit(1)

    # Try to find data in outputDir directly and in immediate sub-directories
    search_dirs = [outputDir] + sorted(
        p for p in outputDir.iterdir() if p.is_dir()
    )
    workingDir = None
    for candidate in search_dirs:
        if list(candidate.glob("statistics_dictionary_qoi_*.pkl")):
            workingDir = candidate
            break

    if workingDir is None:
        print(f"ERROR: No statistics_dictionary_qoi_*.pkl files found in:\n  {outputDir}")
        sys.exit(1)

    if workingDir != outputDir and args.verbose:
        print(f"  Found data in sub-directory: {workingDir.name}")

    # LaTeX rendering for parameter names like $t^+$, $\kappa$ etc.
    if not args.no_latex:
        try:
            matplotlib.rcParams.update({
                "text.usetex": False,         # avoid requiring a TeX installation
                "mathtext.fontset": "dejavusans",
            })
        except Exception:
            pass

    save_dir = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover QoIs ──────────────────────────────────────────────────────
    all_qois = discover_qois(workingDir)
    if not all_qois:
        print(f"ERROR: No statistics_dictionary_qoi_*.pkl found in {workingDir}")
        sys.exit(1)

    qois = [args.qoi] if args.qoi else all_qois
    for q in qois:
        if q not in all_qois:
            print(f"ERROR: QoI '{q}' not found. Available: {all_qois}")
            sys.exit(1)

    print(f"\nOutput directory : {workingDir}")
    print(f"QoIs found       : {all_qois}")
    print(f"Processing       : {qois}")
    if args.top:
        print(f"Top-N parameters : {args.top}")
    if args.clip_negative:
        print(f"Clipping negative Sobol values to 0")

    # ── Load parameter names from df_index_parameter ──────────────────────
    param_names_from_df = get_param_names_from_df(workingDir)
    if args.verbose and param_names_from_df:
        print(f"Parameter names  : {param_names_from_df}")

    # Storage for multi-QoI heatmap
    gen_total_by_qoi: dict[str, dict] = {}
    gen_main_by_qoi:  dict[str, dict] = {}

    # ── Per-QoI processing ─────────────────────────────────────────────────
    for qoi in qois:
        print(f"\n{'═' * 70}")
        print(f"  QoI: {qoi}")
        print(f"{'═' * 70}")

        stats = load_stats_dict(workingDir, qoi)
        if stats is None:
            print(f"  [SKIP] Could not load statistics for {qoi}")
            continue

        # Use parameter names from df_index_parameter; fall back to stats dict
        param_names = param_names_from_df

        # ── Time-wise Sobol ────────────────────────────────────────────────
        df_m = extract_timewise_sobol(stats, param_names, SOBOL_M,
                                      args.clip_negative)
        df_t = extract_timewise_sobol(stats, param_names, SOBOL_T,
                                      args.clip_negative)

        if df_m is not None:
            print_timewise_summary(df_m, "First-order Sobol_m", qoi)
        elif args.verbose:
            print(f"\n  [Sobol_m] not found in statistics dictionary")

        if df_t is not None:
            print_timewise_summary(df_t, "Total-order Sobol_t", qoi)
        elif args.verbose:
            print(f"\n  [Sobol_t] not found in statistics dictionary")

        if not args.no_timewise and (df_m is not None or df_t is not None):
            plot_timewise_sobol(
                df_m, df_t, qoi, args.top, save_dir, args.dpi)

        # ── Generalized Sobol (scalar summary) ────────────────────────────
        gen_total  = extract_generalized_sobol(stats, GEN_TOTAL)
        gen_main   = extract_generalized_sobol(stats, GEN_MAIN)

        if gen_total:
            gen_total = top_n_from_dict(gen_total, args.top)
            print_generalized_summary(gen_total, "Generalized Total-order Sobol", qoi)
        elif args.verbose:
            print(f"\n  [generalized_sobol_total_index] not found")

        if gen_main:
            gen_main = top_n_from_dict(gen_main, args.top)
            print_generalized_summary(gen_main, "Generalized First-order (main) Sobol", qoi)
        elif args.verbose:
            print(f"\n  [generalized_sobol_main_index] not found")

        gen_total_by_qoi[qoi] = gen_total or {}
        gen_main_by_qoi[qoi]  = gen_main  or {}

        if not args.no_generalized and (gen_total or gen_main):
            plot_generalized_sobol_bar(
                gen_total, gen_main, qoi, args.top, save_dir, args.dpi)

        # ── Generalized Sobol over time (if computed) ──────────────────────
        df_gen_tot  = extract_generalized_sobol_over_time(stats, GEN_TOTAL)
        df_gen_main = extract_generalized_sobol_over_time(stats, GEN_MAIN)

        if df_gen_tot is not None:
            print_timewise_summary(df_gen_tot, "Generalized Total-order over time", qoi)
        if df_gen_main is not None:
            print_timewise_summary(df_gen_main, "Generalized First-order over time", qoi)

        if not args.no_generalized and (df_gen_tot is not None or df_gen_main is not None):
            plot_generalized_sobol_over_time(
                df_gen_tot, df_gen_main, qoi, args.top, save_dir, args.dpi)

    # ── Multi-QoI heatmap ─────────────────────────────────────────────────
    if len(qois) > 1:
        plot_heatmap_generalized(
            gen_total_by_qoi, gen_main_by_qoi,
            args.top, save_dir, args.dpi)

    print()


if __name__ == "__main__":
    main()

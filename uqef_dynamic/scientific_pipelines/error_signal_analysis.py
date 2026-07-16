"""
Error Signal Analysis and AR Error Correction
=============================================
Standalone analysis of the model error signal (mean prediction E − observed)
from a completed UQEF-Dynamic UQ run.

Input
-----
df_statistics_and_measured : pd.DataFrame
    Merged statistics + observed + forcing DataFrame produced by
    ``build_statistics()`` in ``hbv_post_analysis.py`` or directly from
    ``statisticsObject`` in the notebook.  Required columns:
      qoi, TimeStamp, E, measured
    Optional columns (used when present):
      StdDev, Var, P10, P90, precipitation, temperature

Usage
-----
From a notebook or another script::

    from uqef_dynamic.scientific_pipelines.error_signal_analysis import run_error_analysis
    df_err, ar_model = run_error_analysis(
        df_statistics_and_measured,
        qoi_column="Q_cms",
        save_dir=pathlib.Path("error_analysis_output"),
    )

As a standalone script (edit the paths at the bottom)::

    python error_signal_analysis.py
"""

import pathlib
import warnings

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sig, stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss, pacf as compute_pacf

from uqef_dynamic.utils import utility

warnings.filterwarnings("ignore")


# =============================================================================
# Internal helpers
# =============================================================================

def _save_mpl(fig, name, save_dir):
    if save_dir is not None:
        fig.savefig(pathlib.Path(save_dir) / f"{name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# =============================================================================
# 1. Build error signal
# =============================================================================

def build_error_signal(df, qoi_column):
    """Extract and annotate the error signal from the merged statistics DataFrame."""
    df_err = (df[df["qoi"] == qoi_column]
              .copy()
              .sort_values(utility.TIME_COLUMN_NAME)
              .reset_index(drop=True))

    df_err["mean_error"] = df_err["E"] - df_err["measured"]

    if "StdDev" not in df_err.columns and "Var" in df_err.columns:
        df_err["StdDev"] = np.sqrt(df_err["Var"])
    if "StdDev" in df_err.columns:
        df_err["error_e_minus_std"] = df_err["mean_error"] - df_err["StdDev"]
        df_err["error_e_plus_std"]  = df_err["mean_error"] + df_err["StdDev"]
    if "P10" in df_err.columns:
        df_err["error_p10"] = df_err["P10"] - df_err["measured"]
        df_err["error_p90"] = df_err["P90"] - df_err["measured"]

    df_err["month"] = pd.to_datetime(df_err[utility.TIME_COLUMN_NAME]).dt.month

    error_ts = df_err.set_index(utility.TIME_COLUMN_NAME)["mean_error"].dropna()
    return df_err, error_ts


def print_summary(error_ts, qoi_column=""):
    label = f" [{qoi_column}]" if qoi_column else ""
    print(f"\n=== Error Signal Summary  "
          f"({len(error_ts)} timesteps, "
          f"{error_ts.index.min()} → {error_ts.index.max()}) ===")
    print(error_ts.describe().to_string())
    print(f"\n  Bias (mean error) : {error_ts.mean():.4f}{label}")
    print(f"  MAE               : {error_ts.abs().mean():.4f}{label}")
    print(f"  RMSE              : {np.sqrt((error_ts**2).mean()):.4f}{label}")
    print(f"  Lag-1  autocorr   : {error_ts.autocorr(1):.3f}")
    print(f"  Lag-7  autocorr   : {error_ts.autocorr(7):.3f}")
    print(f"  Lag-30 autocorr   : {error_ts.autocorr(30):.3f}")


# =============================================================================
# 2. Error analysis plots
# =============================================================================

def plot_error_time_series(df_err, error_ts, qoi_column="", save_dir=None):
    """Time series of the error, rolling statistics, and cumulative sum."""
    time   = df_err[utility.TIME_COLUMN_NAME]
    ylabel = f"Error {qoi_column}" if qoi_column else "Error"
    window = 30

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Raw error + spread bands
    ax = axes[0]
    ax.axhline(0, color="black", lw=0.8, ls="--")
    if "error_e_minus_std" in df_err.columns:
        ax.fill_between(time, df_err["error_e_minus_std"], df_err["error_e_plus_std"],
                        alpha=0.3, color="steelblue", label="Mean ± StdDev")
    if "error_p10" in df_err.columns:
        ax.fill_between(time, df_err["error_p10"], df_err["error_p90"],
                        alpha=0.2, color="orange", label="P10–P90")
    ax.plot(time, df_err["mean_error"], color="steelblue", lw=1.2, label="Mean error (E − obs)")
    ax.set_ylabel(ylabel); ax.set_title("Error signal over time")
    ax.legend(); ax.grid(alpha=0.3)

    # Rolling mean / std
    roll_mean = error_ts.rolling(window, center=True).mean()
    roll_std  = error_ts.rolling(window, center=True).std()
    ax2 = axes[1]
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.fill_between(roll_mean.index, roll_mean - roll_std, roll_mean + roll_std,
                     alpha=0.3, color="tomato")
    ax2.plot(roll_mean.index, roll_mean, color="tomato", lw=1.5,
             label=f"{window}-day rolling mean ± std")
    ax2.set_ylabel(ylabel); ax2.legend(); ax2.grid(alpha=0.3)

    # Cumulative sum (CUSUM) — persistent drift shows as non-returning excursions
    ax3 = axes[2]
    ax3.plot(error_ts.index, error_ts.cumsum(), color="purple", lw=1.2)
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.set_ylabel("Cumulative sum"); ax3.set_xlabel("Date")
    ax3.set_title("CUSUM — persistent drift shows as non-returning excursions")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_time_series", save_dir)


def plot_error_distribution(df_err, error_ts, save_dir=None):
    """Histogram + KDE, monthly boxplot, and empirical CDF."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(error_ts, bins=60, density=True, alpha=0.6, color="steelblue")
    t = np.linspace(error_ts.min(), error_ts.max(), 500)
    kde = cp.GaussianKDE(error_ts.values, h_mat=0.005**2)
    ax.plot(t, kde.pdf(t), color="navy", lw=2, label="KDE")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.axvline(error_ts.mean(), color="orange", ls="--", lw=1,
               label=f"Mean = {error_ts.mean():.2f}")
    ax.set_xlabel("Error"); ax.set_ylabel("Density")
    ax.set_title("Overall Error Distribution"); ax.legend(); ax.grid(alpha=0.3)

    monthly = [df_err[df_err["month"] == m]["mean_error"].dropna().values
               for m in range(1, 13)]
    axes[1].boxplot(monthly, labels=list("JFMAMJJASOND"), patch_artist=True)
    axes[1].axhline(0, color="red", ls="--", lw=1)
    axes[1].set_xlabel("Month"); axes[1].set_ylabel("Error")
    axes[1].set_title("Error by Month (seasonal bias)"); axes[1].grid(alpha=0.3, axis="y")

    sorted_err = np.sort(error_ts)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    axes[2].plot(sorted_err, cdf, color="steelblue", lw=2)
    axes[2].axvline(0, color="red", ls="--", lw=1)
    axes[2].set_xlabel("Error"); axes[2].set_ylabel("Cumulative Probability")
    axes[2].set_title("Empirical CDF of Error"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_distribution", save_dir)


def plot_error_acf_pacf(error_ts, save_dir=None):
    """ACF and PACF up to lag 60."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf( error_ts, lags=60, ax=axes[0], alpha=0.05, title="ACF of Mean Error")
    plot_pacf(error_ts, lags=60, ax=axes[1], alpha=0.05, title="PACF of Mean Error", method="ywm")
    for ax in axes:
        ax.set_xlabel("Lag [days]"); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_mpl(fig, "error_acf_pacf", save_dir)

    print(f"\n  Lag-1  autocorr : {error_ts.autocorr(1):.3f}")
    print(f"  Lag-7  autocorr : {error_ts.autocorr(7):.3f}")
    print(f"  Lag-30 autocorr : {error_ts.autocorr(30):.3f}")
    print(f"  Lag-60 autocorr : {error_ts.autocorr(60):.3f}")


def run_stationarity_tests(error_ts):
    """ADF and KPSS stationarity tests."""
    print("\n=== Stationarity Tests ===")
    err = error_ts.values
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(err, autolag="AIC")
    print(f"  ADF  stat={adf_stat:.4f}  p={adf_p:.4f}  "
          f"→ {'STATIONARY' if adf_p < 0.05 else 'NON-STATIONARY'} at 5%")
    print(f"       critical values: {adf_crit}")
    kpss_stat, kpss_p, _, kpss_crit = kpss(err, regression="c", nlags="auto")
    print(f"  KPSS stat={kpss_stat:.4f}  p={kpss_p:.4f}  "
          f"→ {'NON-STATIONARY' if kpss_p < 0.05 else 'STATIONARY'} at 5%")
    print(f"       critical values: {kpss_crit}")


def plot_error_psd(error_ts, save_dir=None):
    """Welch PSD in frequency and period domain."""
    freqs, psd = sig.welch(error_ts.values, fs=1.0, nperseg=min(len(error_ts) // 4, 256))
    periods = np.where(freqs > 0, 1 / freqs, np.inf)
    markers = [(365, "Annual"), (182, "Semi-annual"), (91, "Quarterly"), (30, "Monthly")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(freqs[1:], psd[1:], color="steelblue")
    for period, label in markers:
        if 1 / period >= freqs[1]:
            axes[0].axvline(1 / period, color="red", ls="--", lw=0.8, alpha=0.7)
            axes[0].text(1 / period, psd[1:].max() * 0.3, label,
                         rotation=90, fontsize=8, color="red")
    axes[0].set_xlabel("Frequency [cycles/day]"); axes[0].set_ylabel("PSD")
    axes[0].set_title("Power Spectral Density (Welch)"); axes[0].grid(alpha=0.3)

    valid = (periods < 400) & (periods > 2)
    axes[1].semilogy(periods[valid], psd[valid], color="tomato")
    for period, label in markers:
        axes[1].axvline(period, color="steelblue", ls="--", lw=0.8, alpha=0.7)
        axes[1].text(period, psd[valid].max() * 0.3, label,
                     rotation=90, fontsize=8, color="steelblue")
    axes[1].set_xlabel("Period [days]"); axes[1].set_ylabel("PSD")
    axes[1].set_title("PSD vs Period"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_psd", save_dir)


def plot_cross_correlation_with_forcing(df_err, save_dir=None):
    """Cross-correlation and scatter of error vs. precipitation / temperature."""
    forcing_cols = [c for c in ["precipitation", "temperature"] if c in df_err.columns]
    if not forcing_cols:
        print("  [skip] No forcing columns found — skipping cross-correlation plot.")
        return

    fig, axes = plt.subplots(2, len(forcing_cols), figsize=(7 * len(forcing_cols), 9))
    if len(forcing_cols) == 1:
        axes = axes.reshape(2, 1)

    for ci, forcing in enumerate(forcing_cols):
        aligned = pd.concat([
            df_err.set_index(utility.TIME_COLUMN_NAME)["mean_error"],
            df_err.set_index(utility.TIME_COLUMN_NAME)[forcing],
        ], axis=1).dropna()
        e, f = aligned["mean_error"], aligned[forcing]
        ci95  = 1.96 / np.sqrt(len(e))
        lags  = list(range(-30, 31))
        xcorr = [e.corr(f.shift(-lag)) for lag in lags]

        axes[0, ci].stem(lags, xcorr, markerfmt="C0o", linefmt="C0-", basefmt="k-")
        axes[0, ci].axhline( ci95, color="red", ls="--", lw=0.8, label="95% CI")
        axes[0, ci].axhline(-ci95, color="red", ls="--", lw=0.8)
        axes[0, ci].set_xlabel("Lag [days]"); axes[0, ci].set_ylabel("Cross-correlation")
        axes[0, ci].set_title(f"Error × {forcing}"); axes[0, ci].legend()
        axes[0, ci].grid(alpha=0.3)

        axes[1, ci].scatter(f, e, alpha=0.3, s=8, color="steelblue")
        m, b, r, p, _ = stats.linregress(f, e)
        x_line = np.linspace(f.min(), f.max(), 100)
        axes[1, ci].plot(x_line, m * x_line + b, color="red", lw=2,
                         label=f"r={r:.3f}  p={p:.3f}")
        axes[1, ci].axhline(0, color="black", lw=0.5, ls="--")
        axes[1, ci].set_xlabel(forcing); axes[1, ci].set_ylabel("Error")
        axes[1, ci].set_title(f"Error vs {forcing}"); axes[1, ci].legend()
        axes[1, ci].grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_vs_forcing", save_dir)


def plot_bias_vs_spread(df_err, save_dir=None):
    """Compare |bias| against UQ spread (StdDev) over time."""
    if "StdDev" not in df_err.columns:
        print("  [skip] StdDev not available — skipping bias-vs-spread plot.")
        return

    time  = df_err[utility.TIME_COLUMN_NAME]
    ratio = df_err["mean_error"].abs() / df_err["StdDev"].replace(0, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(time, df_err["mean_error"].abs(), color="steelblue", lw=1,
                 label="|Mean error| (bias)")
    axes[0].plot(time, df_err["StdDev"], color="tomato", lw=1,
                 label="UQ spread / StdDev (parameter uncertainty)")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Bias vs Parameter Uncertainty over Time")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(time, ratio, color="purple", lw=1, alpha=0.8)
    axes[1].axhline(1.0, color="red", ls="--", lw=1, label="|bias| = spread  (ratio = 1)")
    axes[1].set_ylabel("|Mean error| / StdDev"); axes[1].set_xlabel("Date")
    axes[1].set_title("Ratio > 1: structural error dominates; < 1: spread covers observations")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "error_vs_spread", save_dir)

    pct = (ratio > 1).mean()
    print(f"\n  Timesteps where |bias| > UQ spread: "
          f"{(ratio > 1).sum()} / {len(ratio)}  ({pct:.1%})")


# =============================================================================
# 3. AR order selection
# =============================================================================

def select_ar_order(error_ts, p_max=20, save_dir=None):
    """Determine AR order via PACF cutoff and AIC/BIC/HQIC."""
    ci = 1.96 / np.sqrt(len(error_ts))
    pacf_vals = compute_pacf(error_ts.values, nlags=p_max, method="ywm")

    significant = np.where(np.abs(pacf_vals[1:]) > ci)[0] + 1
    p_pacf = int(significant[-1]) if len(significant) else 1

    records = []
    for p in range(1, p_max + 1):
        res = AutoReg(error_ts.values, lags=p, old_names=False).fit()
        records.append({"p": p, "AIC": res.aic, "BIC": res.bic, "HQIC": res.hqic})
    df_ic  = pd.DataFrame(records).set_index("p")
    p_aic  = int(df_ic["AIC"].idxmin())
    p_bic  = int(df_ic["BIC"].idxmin())
    p_hqic = int(df_ic["HQIC"].idxmin())

    print(f"\n=== AR Order Selection ===")
    print(f"  PACF cutoff → p = {p_pacf}")
    print(f"  AIC         → p = {p_aic}")
    print(f"  BIC         → p = {p_bic}  (used by default)")
    print(f"  HQIC        → p = {p_hqic}")
    print(f"  Significant PACF lags: {significant.tolist()}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    lags = np.arange(p_max + 1)
    axes[0].bar(lags[1:], pacf_vals[1:], color="steelblue", alpha=0.7)
    axes[0].axhline( ci, color="red", ls="--", lw=1, label=f"95% CI (±{ci:.3f})")
    axes[0].axhline(-ci, color="red", ls="--", lw=1)
    axes[0].axhline( 0,  color="black", lw=0.5)
    axes[0].set_xlabel("Lag [days]"); axes[0].set_ylabel("PACF")
    axes[0].set_title(f"PACF  (last significant lag: {p_pacf})")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for col, color, ls in [("AIC","steelblue","-"), ("BIC","tomato","--"), ("HQIC","green",":")]:
        axes[1].plot(df_ic.index, df_ic[col], color=color, ls=ls, lw=1.8, label=col)
        axes[1].axvline(df_ic[col].idxmin(), color=color, lw=0.8, alpha=0.5)
    axes[1].set_xlabel("AR order p"); axes[1].set_ylabel("Information criterion")
    axes[1].set_title("AIC / BIC / HQIC"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plot_acf(error_ts, lags=p_max, ax=axes[2], alpha=0.05,
             title="ACF of error (for reference)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig, "ar_order_selection", save_dir)

    return p_bic


# =============================================================================
# 4. AR(p) fitting, correction, and diagnostics
# =============================================================================

def fit_and_apply_ar_correction(df_err, error_ts, p=1, save_dir=None):
    """Fit AR(p), apply causal one-step-ahead correction, plot diagnostics."""
    model_ar  = AutoReg(error_ts.values, lags=p, old_names=False).fit()
    phi_vec   = model_ar.params[1:]
    intercept = model_ar.params[0]
    sigma_eta = np.std(model_ar.resid)

    print(f"\n=== AR({p}) Fit ===")
    print(f"  Intercept     : {intercept:.4f}")
    print(f"  Coefficients φ: {np.round(phi_vec, 4).tolist()}")
    print(f"  Innovation std σ_η = {sigma_eta:.3f}  "
          f"(σ_ε = {error_ts.std():.3f},  "
          f"variance reduced by {1 - (sigma_eta/error_ts.std())**2:.1%})")

    # Causal correction: at time t, use only ε(t-1)…ε(t-p) which are already observed
    eps    = df_err["mean_error"].values.copy()
    E_corr = df_err["E"].values.copy()
    for t in range(p, len(df_err)):
        eps_past = eps[t - p: t][::-1]          # [ε(t-1), ε(t-2), …, ε(t-p)]
        eps_hat  = intercept + phi_vec @ eps_past
        E_corr[t] = df_err["E"].iloc[t] - eps_hat

    col_E     = f"E_corr_ar{p}"
    col_err   = f"error_corr_ar{p}"
    df_err[col_E]   = E_corr
    df_err[col_err] = E_corr - df_err["measured"]
    err_corr_ts = df_err[col_err].iloc[p:].dropna()

    # --- Residual diagnostics ---
    resid = pd.Series(model_ar.resid)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(resid.values, lw=0.6, color="steelblue")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_title(f"AR({p}) residuals  (should look like white noise)")
    axes[0].grid(alpha=0.3)
    plot_acf( resid, lags=40, ax=axes[1], alpha=0.05,
              title="ACF of residuals  (should be flat)")
    plot_pacf(resid, lags=40, ax=axes[2], alpha=0.05,
              title="PACF of residuals (should be flat)", method="ywm")
    for ax in axes[1:]:
        ax.set_xlabel("Lag [days]"); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_mpl(fig, f"ar{p}_residual_diagnostics", save_dir)

    # --- Comparison: time series + error + rolling MAE ---
    time = df_err[utility.TIME_COLUMN_NAME]
    z    = 1.96

    fig2, axes2 = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    ax = axes2[0]
    if "StdDev" in df_err.columns:
        ax.fill_between(time,
                        df_err["E"] - z * df_err["StdDev"],
                        df_err["E"] + z * df_err["StdDev"],
                        alpha=0.18, color="steelblue",
                        label=f"E ± {z}σ (parameter UQ)")
    ax.fill_between(time,
                    df_err[col_E] - z * sigma_eta,
                    df_err[col_E] + z * sigma_eta,
                    alpha=0.25, color="tomato",
                    label=f"E_corr ± {z}·σ_η  (σ_η = {sigma_eta:.1f})")
    ax.plot(time, df_err["measured"], color="black",     lw=1.3, label="Observed")
    ax.plot(time, df_err["E"],        color="steelblue", lw=1.0, ls="--", alpha=0.8,
            label="E (mean prediction)")
    ax.plot(time, df_err[col_E],      color="tomato",    lw=1.2,
            label=f"E_corr  AR({p})")
    ax.set_ylabel("Streamflow")
    ax.set_title("Observed / Mean prediction / AR-corrected")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)

    ax2 = axes2[1]
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.plot(time, df_err["mean_error"], color="steelblue", lw=0.9, alpha=0.85,
             label=f"Original error   (bias = {df_err['mean_error'].mean():.2f})")
    ax2.plot(time, df_err[col_err],      color="tomato",    lw=0.9, alpha=0.85,
             label=f"Corrected error  (bias = {err_corr_ts.mean():.2f})")
    ax2.set_ylabel("Error")
    ax2.set_title(f"Error: original vs. AR({p})-corrected")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    window = 30
    roll_mae_orig = df_err["mean_error"].abs().rolling(window, center=True).mean()
    roll_mae_corr = df_err[col_err].abs().rolling(window, center=True).mean()
    ax3 = axes2[2]
    ax3.plot(time, roll_mae_orig, color="steelblue", lw=1.5, label="Original  (30-day rolling MAE)")
    ax3.plot(time, roll_mae_corr, color="tomato",    lw=1.5, label="Corrected (30-day rolling MAE)")
    ax3.fill_between(time, roll_mae_orig, roll_mae_corr,
                     where=(roll_mae_orig >= roll_mae_corr),
                     alpha=0.2, color="green", label="AR correction helps")
    ax3.fill_between(time, roll_mae_orig, roll_mae_corr,
                     where=(roll_mae_orig  < roll_mae_corr),
                     alpha=0.2, color="red",   label="AR correction hurts")
    ax3.set_ylabel("30-day rolling MAE"); ax3.set_xlabel("Date")
    ax3.set_title("Rolling MAE: original vs. AR-corrected")
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    plt.tight_layout()
    _save_mpl(fig2, f"ar{p}_correction_comparison", save_dir)

    # --- Summary table ---
    print(f"\n{'Metric':<14} {'Original':>12} {f'AR({p}) corrected':>16}  {'Change':>8}")
    print("─" * 54)
    for label, fo, fc in [
        ("Bias",      error_ts.mean(),                    err_corr_ts.mean()),
        ("MAE",       error_ts.abs().mean(),               err_corr_ts.abs().mean()),
        ("RMSE",      np.sqrt((error_ts**2).mean()),       np.sqrt((err_corr_ts**2).mean())),
        ("Lag-1 ACF", error_ts.autocorr(1),               err_corr_ts.autocorr(1)),
        ("Lag-7 ACF", error_ts.autocorr(7),               err_corr_ts.autocorr(7)),
    ]:
        print(f"{label:<14} {fo:>12.3f} {fc:>16.3f}  {fc - fo:>+8.3f}")

    return df_err, model_ar


# =============================================================================
# Main pipeline
# =============================================================================

def run_error_analysis(
    df_statistics_and_measured,
    qoi_column="Q_cms",
    save_dir=None,
    ar_order=None,
    p_max=20,
):
    """Run the full error signal analysis and AR correction pipeline.

    Parameters
    ----------
    df_statistics_and_measured : pd.DataFrame
        Merged statistics + observed + forcing DataFrame.
    qoi_column : str
        Column name of the quantity of interest (default ``"Q_cms"``).
    save_dir : pathlib.Path or str or None
        Directory to save PDF figures.  ``None`` → figures shown interactively only.
    ar_order : int or None
        AR order for correction.  ``None`` → auto-selected by BIC.
    p_max : int
        Maximum AR order considered during selection (default 20).

    Returns
    -------
    df_err : pd.DataFrame
        Error DataFrame including corrected columns ``E_corr_ar{p}`` and
        ``error_corr_ar{p}``.
    ar_model : statsmodels AutoReg results object
    """
    if save_dir is not None:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build error signal
    df_err, error_ts = build_error_signal(df_statistics_and_measured, qoi_column)
    print_summary(error_ts, qoi_column)

    # 2. Error analysis plots
    print("\nPlotting error analysis...")
    plot_error_time_series(df_err, error_ts, qoi_column, save_dir)
    plot_error_distribution(df_err, error_ts, save_dir)
    plot_error_acf_pacf(error_ts, save_dir)
    run_stationarity_tests(error_ts)
    plot_error_psd(error_ts, save_dir)
    plot_cross_correlation_with_forcing(df_err, save_dir)
    plot_bias_vs_spread(df_err, save_dir)

    # 3. AR order selection + correction
    p = ar_order if ar_order is not None else select_ar_order(error_ts, p_max, save_dir)
    df_err, ar_model = fit_and_apply_ar_correction(df_err, error_ts, p, save_dir)

    if save_dir:
        print(f"\nDone. Figures saved to {save_dir}")
    return df_err, ar_model


if __name__ == "__main__":
    # Edit these paths and run directly:  python error_signal_analysis.py
    import pickle
    df = pd.read_pickle("df_statistics_and_measured.pkl")
    run_error_analysis(
        df,
        qoi_column="Q_cms",
        save_dir=pathlib.Path("error_analysis_output"),
    )

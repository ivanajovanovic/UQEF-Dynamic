"""
extract_timing_info.py
======================
Reads a UQEF-Dynamic simulation output directory and extracts all timing
information that the framework tracks and saves across its different layers:

  Layer 1 - time_info.txt          (written by every pipeline script)
  Layer 2 - uqsim_args.pkl         (run configuration context)
  Layer 3 -dict_with_time_info.txt (written by uqef_dynamic_postprocessing.py,
                                     looked for in <workingDir> and any
                                     immediate sub-directories)

Usage
-----
    python extract_timing_info.py <workingDir> [--verbose] [--output FILE]

Example
-------
    python extract_timing_info.py debug_output/battery_mc_70000_kl_local_debug
    python extract_timing_info.py debug_output/battery_mc_70000_kl_local_debug --verbose
    python extract_timing_info.py debug_output/battery_mc_70000_kl_local_debug --output timing_report.txt
"""

import argparse
import pathlib
import pickle
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Tee: write to stdout and a file simultaneously
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    """Mirrors writes to sys.stdout and an open file handle."""
    def __init__(self, file_handle):
        self._file = file_handle
        self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_key_value_txt(path: pathlib.Path) -> dict:
    """Read a simple 'key: value\\n' text file into a dict."""
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            try:
                result[key.strip()] = float(val.strip())
            except ValueError:
                result[key.strip()] = val.strip()
    return result


def _fmt(val, unit="s") -> str:
    """Format a numeric timing value with sensible units."""
    if val is None:
        return "n/a"
    if isinstance(val, str):
        return val
    if unit == "s":
        if val >= 3600:
            return f"{val:>12.2f} s  ({val/3600:.2f} h)"
        if val >= 60:
            return f"{val:>12.2f} s  ({val/60:.2f} min)"
        return f"{val:>12.4f} s"
    return f"{val}"


def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 – time_info.txt
# ─────────────────────────────────────────────────────────────────────────────

def extract_time_info(workingDir: pathlib.Path, verbose: bool):
    """
    Reads time_info.txt.

    Keys (note not all of these are always present):
        number_full_model_runs - total samples evaluated
        time_model_simulations - wall time for uqsim.simulate()
        time_computing_statistics -wall time for prepare + calc statistics
        total_time - end-to-end wall time of the script
        mean_model_eval_time - mean pure model call time per sample
        total_model_eval_time - sum of all pure model call times
    """
    path = workingDir / "time_info.txt"
    if not path.exists():
        print(f"  [MISSING] {path}")
        return {}

    d = _read_key_value_txt(path)
    _section("Layer 1 – time_info.txt  (pipeline wall times)")

    n = d.get("number_full_model_runs")
    print(f"  {'number_full_model_runs':<38} {int(n) if n else 'n/a'}")

    for key in [
        "time_model_simulations",
        "time_computing_statistics",
        "total_time",
        "mean_model_eval_time",
        "total_model_eval_time",
    ]:
        val = d.get(key)
        label = key
        if val is not None:
            print(f"  {label:<38} {_fmt(val)}")
        elif verbose:
            print(f"  {label:<38} [not recorded in this run]")

    # Derived: mean per-sample time from the outer timer
    t_sim = d.get("time_model_simulations")
    if t_sim and n:
        mean_outer = float(t_sim) / float(n)
        print(f"  {'derived: mean outer time/sample':<38} {_fmt(mean_outer)}"
              f"  (time_model_simulations / n_runs)")

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 – uqsim_args.pkl  (configuration context)
# ─────────────────────────────────────────────────────────────────────────────

def extract_args_context(workingDir: pathlib.Path, verbose: bool):
    """
    Reads uqsim_args.pkl for the run configuration that contextualises timing.
    Does not contain timing itself, but provides useful metadata.
    """
    path = workingDir / "uqsim_args.pkl"
    if not path.exists():
        if verbose:
            print(f"\n  [MISSING] {path}")
        return

    _section("Layer 2 – uqsim_args.pkl  (run configuration context)")
    with open(path, "rb") as f:
        args = pickle.load(f)

    fields = [
        ("model",                 "model"),
        ("uq_method",             "uq_method"),
        ("mc_numevaluations",     "mc_numevaluations"),
        ("sampling_rule",         "sampling_rule"),
        ("mpi",                   "mpi"),
        ("mpi_method",            "mpi_method"),
        ("parallel_statistics",   "parallel_statistics"),
        ("regression",            "regression"),
        ("regression_model_type", "regression_model_type"),
        ("sc_p_order",            "sc_p_order"),
        ("kl_expansion_order",    "kl_expansion_order"),
        ("compute_kl_expansion_of_qoi",        "compute_kl_expansion_of_qoi"),
        ("compute_generalized_sobol_indices",   "compute_generalized_sobol_indices"),
        ("save_gpce_surrogate",                 "save_gpce_surrogate"),
    ]

    for label, attr in fields:
        val = getattr(args, attr, "[not set]")
        print(f"  {label:<38} {val}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 – dict_with_time_info.txt  (postprocessing timing)
# ─────────────────────────────────────────────────────────────────────────────

def extract_postprocessing_timing(workingDir: pathlib.Path, verbose: bool):
    """
    Looks for dict_with_time_info.txt written by uqef_dynamic_postprocessing.py.
    Searched in workingDir itself and all immediate sub-directories.

    Keys that may be present:
        time_reading_all_saved_data             – loading pickled results
        time_paralle_statistics_recomputation   – re-running statistics
        time_parallel_original_model_reevaluations – re-running DFN model
        mean_original_model_eval_time_per_sample   – mean DFN eval time per sample
        time_parallel_pce_surrogate_reevaluations  – re-evaluating PCE surrogate
        time_kl_surrogate_reevaluations            – re-evaluating KL+PCE surrogate
        mean_surrogate_eval_time_per_sample        – mean surrogate eval time
        time_paralle_analysing_pce_surrogate       – analysing PCE coefficients
        time_generalized_si_recomputation          – recomputing generalised SI
        total_runtime                              – full postprocessing wall time
    """
    candidates = [workingDir / "dict_with_time_info.txt"] + \
                 sorted(workingDir.glob("*/dict_with_time_info.txt"))

    found = [p for p in candidates if p.exists()]
    if not found:
        if verbose:
            print(f"\n  [MISSING] dict_with_time_info.txt  "
                  f"(postprocessing has not been run yet)")
        return

    for path in found:
        rel = path.relative_to(workingDir)
        _section(f"Layer 3 – {rel}  (postprocessing wall times)")
        d = _read_key_value_txt(path)

        ordered_keys = [
            "time_reading_all_saved_data",
            "time_paralle_statistics_recomputation",
            "time_parallel_original_model_reevaluations",
            "mean_original_model_eval_time_per_sample",
            "time_parallel_pce_surrogate_reevaluations",
            "time_kl_surrogate_reevaluations",
            "mean_surrogate_eval_time_per_sample",
            "time_paralle_analysing_pce_surrogate",
            "time_generalized_si_recomputation",
            "total_runtime",
        ]
        for key in ordered_keys:
            val = d.get(key)
            if val is not None:
                print(f"  {key:<44} {_fmt(val)}")
            elif verbose:
                print(f"  {key:<44} [not recorded]")

        # any extra keys not in the known list
        extra = {k: v for k, v in d.items() if k not in ordered_keys}
        if extra:
            print(f"\n  Additional keys:")
            for k, v in extra.items():
                print(f"    {k:<42} {_fmt(v)}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(time_info: dict):
    n          = time_info.get("number_full_model_runs")
    t_sim      = time_info.get("time_model_simulations")
    t_stats    = time_info.get("time_computing_statistics")
    t_total    = time_info.get("total_time")
    t_mean_eval= time_info.get("mean_model_eval_time")
    t_tot_eval = time_info.get("total_model_eval_time")

    _section("Summary")
    rows = [
        ("Samples (N)",             f"{int(n)}" if n else "n/a"),
        ("Simulation wall time",    _fmt(t_sim)),
        ("Statistics wall time",    _fmt(t_stats)),
        ("Total script wall time",  _fmt(t_total)),
    ]
    if t_sim and n:
        rows.append(("  mean outer time / sample", _fmt(float(t_sim) / float(n))))
    if t_mean_eval:
        rows.append(("  mean pure model call / sample", _fmt(t_mean_eval)))
    if t_tot_eval and t_sim:
        overhead = float(t_sim) - float(t_tot_eval)
        rows.append(("  MPI/IO overhead in simulate()", _fmt(overhead)))

    for label, val in rows:
        print(f"  {label:<38} {val}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract all timing information from a UQEF-Dynamic output directory.")
    parser.add_argument("workingDir", type=pathlib.Path,
                        help="Path to the simulation output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Also print keys that are missing / not recorded")
    parser.add_argument(
        "--output", "-o", type=pathlib.Path, default=None,
        help="Path of the .txt file to save the report to "
             "(default: <workingDir>/timing_report.txt)")
    args = parser.parse_args()

    workingDir = args.workingDir.resolve()
    if not workingDir.exists():
        print(f"ERROR: directory does not exist: {workingDir}")
        sys.exit(1)

    output_path = args.output if args.output is not None \
        else workingDir / "timing_report.txt"
    output_path = output_path.resolve()

    with open(output_path, "w") as fh:
        sys.stdout = _Tee(fh)
        try:
            print(f"\nExtracting timing info from:\n  {workingDir}\n")
            print(f"Report saved to:\n  {output_path}\n")

            time_info = extract_time_info(workingDir, args.verbose)
            extract_args_context(workingDir, args.verbose)
            extract_postprocessing_timing(workingDir, args.verbose)
            if time_info:
                print_summary(time_info)
            print()
        finally:
            sys.stdout = sys.stdout._stdout  # restore even on error

    # confirmation goes only to the real stdout (already restored)
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()

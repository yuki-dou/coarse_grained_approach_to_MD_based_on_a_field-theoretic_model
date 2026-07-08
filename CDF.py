import json
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import argparse

np.random.seed(42)


def logistic_decay_log(T, T50, k):
    """
    Logistic decay function in logarithmic temperature scale.

    f(T) = 100 / (1 + (T/T50)^k)

    Parameters
    ----------
    T : array_like
        Temperature values.
    T50 : float
        Midpoint temperature where f(T50) = 50.
    k : float
        Steepness parameter in log-space.

    Returns
    -------
    array_like
        Function values in [0, 100].
    """
    T = np.clip(T, 1e-30, None)
    ratio = T / T50
    z = k * np.log10(ratio)
    z = np.clip(z, -500, 500)
    return 100.0 / (1.0 + np.exp(z))


def metrics_from_params(T50, k):
    """
    Compute T10, T90, and dT from logistic parameters.

    For logistic in log-space:
        T90 = T50 * 10^(ln(9)/k)
        T10 = T50 * 10^(-ln(9)/k)

    Parameters
    ----------
    T50 : float
        Midpoint temperature.
    k : float
        Steepness parameter.

    Returns
    -------
    tuple
        (T10, T50, T90, dT)
    """
    if k <= 0:
        return T50, T50, T50, 0.0
    delta_log = np.log(9.0) / k
    T90 = T50 * (10.0 ** delta_log)
    T10 = T50 * (10.0 ** (-delta_log))
    dT = T90 - T10
    return T10, T50, T90, dT


def estimate_initial_params(temps, counts):
    """
    Estimate initial parameters T50 and k for logistic fit.

    Parameters
    ----------
    temps : ndarray
        Temperature values.
    counts : ndarray
        Population counts.

    Returns
    -------
    tuple
        (T50_init, k_init)
    """
    max_val = np.max(counts)
    y_norm = (counts / max_val) * 100.0
    idx_mid = np.argmin(np.abs(y_norm - 50))
    T50_init = temps[idx_mid]

    if 0 < idx_mid < len(temps) - 1:
        dy = y_norm[idx_mid + 1] - y_norm[idx_mid - 1]
        dT = temps[idx_mid + 1] - temps[idx_mid - 1]
        if abs(dT) > 1e-12:
            df_dT = dy / dT
            df_dlogT = df_dT * temps[idx_mid] * np.log(10)
            k_init = abs(df_dlogT) / 25.0
            k_init = np.clip(k_init, 1.0, 100.0)
        else:
            k_init = 10.0
    else:
        k_init = 10.0
    return T50_init, k_init


def bootstrap_logistic(temps, counts, n_boot=5000):
    """
    Bootstrap uncertainty estimation for logistic fit parameters.

    Parameters
    ----------
    temps : ndarray
        Temperature values.
    counts : ndarray
        Population counts.
    n_boot : int
        Number of bootstrap samples.

    Returns
    -------
    dict or None
        Dictionary with T10, T50, T90, dT means and standard deviations,
        plus fitted T50 and k. Returns None if fitting fails.
    """
    max_val = np.max(counts)
    if max_val <= 0:
        return None

    y_norm = (counts / max_val) * 100.0
    T50_init, k_init = estimate_initial_params(temps, counts)
    p0 = [T50_init, k_init]
    bounds = ([np.min(temps) * 0.5, 0.5], [np.max(temps) * 2.0, 100.0])

    boot_results = []
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        indices = rng.integers(0, len(temps), len(temps))
        t_b = temps[indices]
        y_b = y_norm[indices]

        sort_idx = np.argsort(t_b)
        t_b = t_b[sort_idx]
        y_b = y_b[sort_idx]

        try:
            popt, _ = curve_fit(logistic_decay_log, t_b, y_b,p0=p0, bounds=bounds, maxfev=5000)
            T50_fit, k_fit = popt
            T10, _, T90, dT = metrics_from_params(T50_fit, k_fit)
            boot_results.append((T10, T50_fit, T90, dT, T50_fit, k_fit))
        except Exception:
            continue

    if len(boot_results) < 10:
        return None

    boot_arr = np.array(boot_results)
    means = np.mean(boot_arr, axis=0)
    stds = np.std(boot_arr, axis=0)

    return {
        'T10': (means[0], stds[0]),
        'T50': (means[1], stds[1]),
        'T90': (means[2], stds[2]),
        'dT': (means[3], stds[3]),
        'T50_fit': (means[4], stds[4]),
        'k_fit': (means[5], stds[5])
    }


def extract_transition_metrics(input_file):
    """
    Extract logistic transition metrics for knot 5_2 from aggregated JSON data.

    Parameters
    ----------
    input_file : str
        Path to JSON file with aggregated simulation data.

    Returns
    -------
    list
        Table rows with [steps, T10, T90, dT, T50] formatted strings.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    steps = sorted(data.keys(), key=lambda x: float(x.replace('m', '')))
    table_rows = []

    for step in steps:
        records = {}
        step_data = data[step]

        for idx, record in step_data.items():
            if 'avg_temperature' in record and record['avg_temperature'] is not None:
                temp = float(record['avg_temperature'])
                if temp > 0:
                    records[int(idx)] = {
                        'temperature': temp,
                        'knots': record.get('knots', {})
                    }

        if not records:
            continue

        sorted_indices = sorted(records.keys())
        temps = np.array([records[i]['temperature'] for i in sorted_indices])
        counts = np.array([records[i]['knots'].get('5_2', 0) for i in sorted_indices], dtype=float)

        if np.max(counts) <= 10 or np.min(counts) >= np.max(counts) * 0.9:
            continue

        result = bootstrap_logistic(temps, counts, n_boot=5000)
        if result is None:
            continue

        step_display = step.replace('m', '')
        table_rows.append([
            step_display,
            f"{result['T10'][0]:.4f} ± {result['T10'][1]:.4f}",
            f"{result['T90'][0]:.4f} ± {result['T90'][1]:.4f}",
            f"{result['dT'][0]:.4f} ± {result['dT'][1]:.4f}",
            f"{result['T50'][0]:.4f} ± {result['T50'][1]:.4f}"
        ])
    return table_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, default='data/knots-data.json')
    parser.add_argument('-o', '--output', type=Path, default='results/cdf-results.txt')
    args = parser.parse_args()

    table = extract_transition_metrics(args.input)

    header = f"{'Steps':<12}{'T_10 ± δ':<22}{'T_90 ± δ':<22}{'ΔT ± δ':<22}{'T_50 ± δ':<22}"
    lines = [header]
    for row in table:
        lines.append(f"{row[0]:<12}{row[1]:<22}{row[2]:<22}{row[3]:<22}{row[4]:<22}")
    output_text = '\n'.join(lines)
    print(output_text)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
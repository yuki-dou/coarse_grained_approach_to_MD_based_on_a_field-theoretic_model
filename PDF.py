import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import argparse

np.random.seed(42)


def extract_unfolding_temps(json_file):
    """
    Extract individual unknotting temperatures from aggregated simulation data.

    For each simulation mode, tracks the decline of knot 5_2 population
    across temperature steps. Each time the population drops, the temperature
    is recorded as many times as the count decreased, reconstructing
    individual unknotting events.

    Parameters
    ----------
    json_file : str
        Path to aggregated JSON data file.

    Returns
    -------
    dict
        Mapping of mode names to arrays of unknotting temperatures.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_temps = {}
    for mode, steps in data.items():
        try:
            mode_val = float(mode.strip().replace('m', '')) * 1e6
            mode_name = f"{int(mode_val / 1e6)}M"
        except ValueError:
            continue

        temps = []
        sorted_steps = sorted(steps.items(), key=lambda x: int(x[0]))
        prev_count = 100

        for _, record in sorted_steps:
            count = record['knots'].get('5_2', 0)
            temp = record['avg_temperature']
            if count < prev_count:
                diff = prev_count - count
                temps.extend([temp] * diff)
            prev_count = count

        all_temps[mode_name] = np.array(temps)
    return all_temps


def logistic_cdf(T, T50, width):
    """
    Cumulative distribution function of the logistic distribution.

    F(T) = 1 / (1 + exp(-(T - T50) / width))

    Parameters
    ----------
    T : array_like
        Temperature values.
    T50 : float
        Location parameter (median).
    width : float
        Scale parameter.

    Returns
    -------
    array_like
        CDF values in [0, 1].
    """
    z = (T - T50) / width
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_pdf(T, T50, width):
    """
    Probability density function of the logistic distribution.

    f(T) = (1/width) * exp(-z) / (1 + exp(-z))^2, where z = (T - T50)/width

    Parameters
    ----------
    T : array_like
        Temperature values.
    T50 : float
        Location parameter (median).
    width : float
        Scale parameter.

    Returns
    -------
    array_like
        PDF values.
    """
    z = (T - T50) / width
    z = np.clip(z, -500, 500)
    exp_z = np.exp(-z)
    return (1.0 / width) * exp_z / (1.0 + exp_z) ** 2


def estimate_width(temps):
    """
    Estimate logistic scale parameter from IQR.

    For logistic distribution: IQR = 2 * ln(3) * width.

    Parameters
    ----------
    temps : ndarray
        Temperature values.

    Returns
    -------
    float
        Estimated width parameter.
    """
    iqr = np.percentile(temps, 75) - np.percentile(temps, 25)
    width = iqr / (2.0 * np.log(3.0))
    if width <= 0:
        width = max((np.max(temps) - np.min(temps)) / 10.0, 1e-6)
    return width


def fit_logistic_to_raw_temps(temps, n_boot=2000):
    """
    Fit logistic distribution to raw unknotting temperatures with bootstrap error.

    Parameters
    ----------
    temps : ndarray
        Array of unknotting temperatures.
    n_boot : int
        Number of bootstrap samples.

    Returns
    -------
    dict or None
        Dictionary with T50, width, T10, T90, dT, and optionally T50_err.
        Returns None if fitting fails.
    """
    if len(temps) < 5:
        return None

    sorted_temps = np.sort(temps)
    n = len(sorted_temps)
    cdf = np.arange(1, n + 1) / n  # empirical CDF

    T50_init = np.median(sorted_temps)
    width_init = estimate_width(sorted_temps)

    p0 = [T50_init, width_init]
    bounds = (
        [np.min(sorted_temps), 1e-12],
        [np.max(sorted_temps), max(np.ptp(sorted_temps), 0.01)]
    )

    try:
        popt, _ = curve_fit(logistic_cdf, sorted_temps, cdf, p0=p0, bounds=bounds, maxfev=5000)
        T50_opt, width_opt = popt
    except Exception:
        return None

    ln9 = np.log(9.0)
    T10 = T50_opt - width_opt * ln9
    T90 = T50_opt + width_opt * ln9
    dT = T90 - T10

    result = {
        'T50': T50_opt,
        'width': width_opt,
        'T10': T10,
        'T90': T90,
        'dT': dT,
    }

    boot_T50 = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        indices = rng.integers(0, n, n)
        t_boot = sorted_temps[indices]
        t_boot = np.sort(t_boot)
        cdf_boot = np.arange(1, n + 1) / n

        try:
            popt_b, _ = curve_fit(logistic_cdf, t_boot, cdf_boot, p0=p0, bounds=bounds, maxfev=2000)
            boot_T50.append(popt_b[0])
        except Exception:
            continue

    if len(boot_T50) > 10:
        result['T50_err'] = np.std(boot_T50)
    else:
        result['T50_err'] = np.nan
    return result


def compute_summary_table(temps_data, modes=None):
    """
    Compute summary statistics table for unknotting temperatures.

    Parameters
    ----------
    temps_data : dict
        Mapping of mode names to temperature arrays.
    modes : list or None
        Modes to include. If None, uses all available modes.

    Returns
    -------
    list
        Table rows with [mode, N, T50_median, T50_fit, T50_err, mean, std].
    """
    if modes is None:
        modes = sorted(temps_data.keys(), key=lambda x: int(x.replace('M', '')))

    table = []
    for mode in modes:
        if mode not in temps_data:
            continue
        temps = temps_data[mode]
        n = len(temps)
        median = np.median(temps)
        mean = np.mean(temps)
        std = np.std(temps)

        fit = fit_logistic_to_raw_temps(temps, n_boot=2000)
        if fit is not None and not np.isnan(fit.get('T50_err', np.nan)):
            T50_fit = fit['T50']
            T50_err = fit['T50_err']
        else:
            T50_fit = np.nan
            T50_err = np.nan

        table.append([mode, n, median, T50_fit, T50_err, mean, std])
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, default='data/knots-data.json')
    parser.add_argument('-o', '--output', type=Path, default='results/pdf-results.txt')
    parser.add_argument('-m', '--modes', nargs='+', default=None, help='Modes to include (e.g. 15M 20M 25M)')
    args = parser.parse_args()

    temps_data = extract_unfolding_temps(args.input)
    table = compute_summary_table(temps_data, args.modes)

    header = (
        f"{'Mode':<10}{'N':<8}{'T50 (med)':<14}"
        f"{'T50 (fit)':<14}{'σ_fit':<14}{'Mean':<14}{'Std':<14}"
    )
    lines = [header]
    for row in table:
        T50_fit_str = f"{row[3]:.5f}" if not np.isnan(row[3]) else "N/A"
        T50_err_str = f"{row[4]:.5f}" if not np.isnan(row[4]) else "N/A"
        lines.append(
            f"{row[0]:<10}{row[1]:<8}{row[2]:<14.5f}"
            f"{T50_fit_str:<14}{T50_err_str:<14}{row[5]:<14.5f}{row[6]:<14.5f}"
        )

    output_text = '\n'.join(lines)
    print(output_text)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
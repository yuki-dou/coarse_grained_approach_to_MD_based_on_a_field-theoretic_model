import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import argparse


def linear(x, a, b):
    """
    Linear function for WLS extrapolation.

    f(x) = a * x + b

    Parameters
    ----------
    x : array_like
        Independent variable (1/N).
    a : float
        Slope.
    b : float
        Intercept (T_eq).

    Returns
    -------
    array_like
        Linear function values.
    """
    return a * x + b


def parse_table(file_path):
    """
    Parse transition metrics table from text file.

    Parameters
    ----------
    file_path : Path
        Path to results table.

    Returns
    -------
    tuple
        (N, T50, errors) arrays extracted from table.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    N_list = []
    T50_list = []
    err_list = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('Steps'):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        try:
            steps = int(parts[0])
            N_list.append(steps * 1e6)

            for i, part in enumerate(parts):
                if part == '±' and i >= 2:
                    if i == 2:
                        T50_idx = None
                        count = 0
                        for j, p in enumerate(parts):
                            if p == '±':
                                count += 1
                                if count == 4:
                                    T50_idx = j - 1
                                    break
                        if T50_idx is not None:
                            T50_list.append(float(parts[T50_idx]))
                            err_list.append(float(parts[T50_idx + 2]))
                    break
            else:
                T50_list.append(float(parts[8]))
                err_list.append(float(parts[10]))

        except (ValueError, IndexError):
            continue

    return np.array(N_list), np.array(T50_list), np.array(err_list)


def compute_extrapolation(N, T50, errors):
    """
    Compute T_eq via weighted least squares extrapolation of T50 vs 1/N.

    Excludes plateau points (N >= 30e6) and the outlier at 20e6.

    Parameters
    ----------
    N : ndarray
        Number of steps.
    T50 : ndarray
        Midpoint temperatures.
    errors : ndarray
        Uncertainties on T50.

    Returns
    -------
    dict
        Dictionary with T_eq, T_eq_err, R2, slope, and plateau value.
    """
    X = 1.0 / N

    mask = (N != 20e6) & (N < 30e6)
    X_fit = X[mask]
    T50_fit = T50[mask]
    err_fit = errors[mask]

    if len(X_fit) == 0:
        raise ValueError(
            "No data points remaining after excluding N=20e6 and N>=30e6. "
            f"Available N: {N.tolist()}"
        )

    popt, pcov = curve_fit(linear, X_fit, T50_fit, sigma=err_fit, absolute_sigma=True)
    slope, intercept = popt
    intercept_err = np.sqrt(pcov[1, 1])

    y_pred = linear(X_fit, *popt)
    weights = 1.0 / (err_fit ** 2)
    ss_res = np.sum(weights * (T50_fit - y_pred) ** 2)
    ss_tot = np.sum(weights * (T50_fit - np.mean(T50_fit)) ** 2)
    R2 = 1 - ss_res / ss_tot

    plateau_mask = N >= 30e6
    plateau_value = np.mean(T50[plateau_mask]) if np.any(plateau_mask) else np.nan

    return {
        'T_eq': intercept,
        'T_eq_err': intercept_err,
        'R2': R2,
        'slope': slope,
        'plateau': plateau_value,
        'points_used': np.sum(mask),
        'points_total': len(N),
        'fitted_N': N[mask].tolist()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, default='results/cdf-results.txt')
    parser.add_argument('-o', '--output', type=Path, default='results/extrapolation-results.txt')
    args = parser.parse_args()

    N, T50, errors = parse_table(args.input)
    result = compute_extrapolation(N, T50, errors)

    lines = [
        f"T_eq = {result['T_eq']:.5f} ± {result['T_eq_err']:.5f}",
        f"R²   = {result['R2']:.4f}",
        f"Plateau (N ≥ 30M) = {result['plateau']:.5f}",
        f"Points used / total = {result['points_used']} / {result['points_total']}",
        f"Fitted N = {result['fitted_N']}"
    ]
    output_text = '\n'.join(lines)
    print(output_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(output_text)
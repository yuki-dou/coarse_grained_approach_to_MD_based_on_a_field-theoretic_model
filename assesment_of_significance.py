import json
import numpy as np
from scipy.stats import norm, binomtest
from pathlib import Path
import argparse


def wilson_ci(count, n_total, alpha=0.05):
    """
    Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    count : int
        Number of successes.
    n_total : int
        Total number of trials.
    alpha : float
        Significance level.

    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if n_total == 0:
        return (0.0, 0.0)

    p_hat = count / n_total
    z = norm.ppf(1 - alpha / 2)

    denominator = 1 + z ** 2 / n_total
    center = (p_hat + z ** 2 / (2 * n_total)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z ** 2 / (4 * n_total ** 2)) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def analyze_knot_intermediacy(json_file, knot_type='3_1', p_noise=0.02, epsilon=0.10, alpha=0.05):
    """
    Test whether a knot type appears as a statistically significant intermediate state.

    For each simulation mode, tests the null hypothesis that the knot population
    is noise (binomial with probability p_noise) at each temperature step.
    Uses Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    json_file : str
        Path to aggregated JSON data file.
    knot_type : str
        Knot type to analyze.
    p_noise : float
        Null hypothesis probability for noise.
    epsilon : float
        Threshold for meaningful population fraction.
    alpha : float
        Significance level (before Bonferroni correction).

    Returns
    -------
    list
        Summary table rows with [mode, n_points, max_fraction, max_upper_ci,
        significant_points, verdict].
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    summary = []

    for mode in sorted(data.keys(), key=lambda x: float(x.replace('m', ''))):
        mode_data = data[mode]

        temps = []
        counts = []
        total_configs = None

        for step in sorted(mode_data.keys(), key=int):
            record = mode_data[step]
            if 'avg_temperature' not in record:
                continue
            knots = record.get('knots', {})
            count = knots.get(knot_type, 0)
            total = sum(knots.values())

            if total_configs is None:
                total_configs = total

            temps.append(record['avg_temperature'])
            counts.append(count)

        if total_configs is None or total_configs == 0:
            continue

        temps = np.array(temps)
        counts = np.array(counts)

        n_points = len(temps)
        if n_points == 0:
            continue

        bonferroni_alpha = alpha / n_points

        significant_points = 0
        max_upper = 0.0
        max_fraction = 0.0

        for j in range(n_points):
            n_knot = int(counts[j])
            low, up = wilson_ci(n_knot, total_configs, alpha)
            result = binomtest(n_knot, total_configs, p=p_noise, alternative='greater')
            p_val = result.pvalue

            is_signif = (p_val < bonferroni_alpha) and (low > 0)
            if is_signif:
                significant_points += 1

            max_upper = max(max_upper, up)
            max_fraction = max(max_fraction, n_knot / total_configs)

        reasons = []
        if significant_points == 0:
            reasons.append("no significant deviation from noise")
        if max_upper < epsilon:
            reasons.append(f"max upper CI ({max_upper:.4f}) below epsilon = {epsilon}")
        if max_fraction < 0.05:
            reasons.append(f"max fraction ({max_fraction:.4f}) < 5%")

        verdict = "noise" if reasons else "possible"
        mode_display = mode.replace('m', 'M')

        summary.append({
            'mode': mode_display,
            'n_points': n_points,
            'max_fraction': max_fraction,
            'max_upper_ci': max_upper,
            'significant_points': significant_points,
            'verdict': verdict
        })

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, default='data/knots-data.json')
    parser.add_argument('-o', '--output', type=Path, default='results/significance-results.txt')
    parser.add_argument('-k', '--knot-type', type=str, default='3_1', help='Knot type to analyze')
    parser.add_argument('-p', '--p-noise', type=float, default=0.02, help='Null hypothesis noise probability')
    parser.add_argument('-e', '--epsilon', type=float, default=0.10, help='Threshold for meaningful population')
    parser.add_argument('-a', '--alpha', type=float, default=0.05, help='Significance level')
    args = parser.parse_args()

    summary = analyze_knot_intermediacy(args.input, knot_type=args.knot_type, p_noise=args.p_noise, epsilon=args.epsilon, alpha=args.alpha)

    header = (
        f"{'Mode':<12}{'Points':<10}{'Max frac':<12}"
        f"{'Max upper CI':<14}{'Signif':<10}{'Verdict':<12}"
    )
    lines = [header]
    for row in summary:
        lines.append(
            f"{row['mode']:<12}{row['n_points']:<10}{row['max_fraction']:<12.4f}"
            f"{row['max_upper_ci']:<14.4f}{row['significant_points']:<10}{row['verdict']:<12}"
        )

    output_text = '\n'.join(lines)
    print(output_text)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
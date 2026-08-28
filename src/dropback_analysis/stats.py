import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def chi_square_test(df, feature, target='isDropback'):
    contingency = pd.crosstab(df[feature], df[target])
    chi2, p_value, dof, expected = chi2_contingency(contingency, correction=False)
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

    # per-category contribution to the overall chi2 statistic
    cell_contributions = (contingency - expected_df) ** 2 / expected_df
    by_category = cell_contributions.sum(axis=1).sort_values(ascending=False)

    # cramer's v effect size: 0 = no association, 1 = perfect association
    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else float('nan')

    print(f"{feature} vs {target}")
    print(f"p-value: {p_value:.2e}")
    print(f"cramer's v: {cramers_v:.3f}")

    if p_value < 0.05:
        print(f"significant (p < 0.05) - {feature} and {target} are not independent")
    else:
        print(f"not significant - no evidence {feature} affects {target}")

    print(f"\nchi2 contribution by {feature}:")
    print((by_category / chi2 * 100).round(1).astype(str) + '%')
    print()

    return {
        'feature': feature,
        'chi2': chi2,
        'dof': dof,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'contingency': contingency,
        'by_category': by_category,
    }

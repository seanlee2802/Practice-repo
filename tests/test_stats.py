import pandas as pd
import pytest

from dropback_analysis.stats import chi_square_test


def test_chi_square_test_perfect_independence():
    # 'group' has no bearing on 'outcome' at all - both groups split 50/50
    df = pd.DataFrame({
        'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'outcome': [True, True, False, False, True, True, False, False],
    })

    result = chi_square_test(df, 'group', target='outcome')

    assert result['chi2'] == pytest.approx(0)
    assert result['p_value'] == pytest.approx(1.0)
    assert result['cramers_v'] == pytest.approx(0)


def test_chi_square_test_perfect_association():
    # 'group' fully determines 'outcome' - A is always True, B is always False
    df = pd.DataFrame({
        'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'outcome': [True, True, True, True, False, False, False, False],
    })

    result = chi_square_test(df, 'group', target='outcome')

    assert result['p_value'] < 0.05
    assert result['cramers_v'] == pytest.approx(1.0)


def test_chi_square_test_by_category_sums_to_chi2():
    # the per-category breakdown is only meaningful if it adds back up to the overall statistic
    df = pd.DataFrame({
        'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'outcome': [True, True, True, True, False, False, False, False],
    })

    result = chi_square_test(df, 'group', target='outcome')

    assert result['by_category'].sum() == pytest.approx(result['chi2'])


def test_chi_square_test_return_keys():
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'outcome': [True, False, True, False],
    })

    result = chi_square_test(df, 'group', target='outcome')

    assert set(result.keys()) == {
        'feature', 'chi2', 'dof', 'p_value', 'cramers_v', 'contingency', 'by_category',
    }
    assert result['feature'] == 'group'
    assert result['dof'] == 1

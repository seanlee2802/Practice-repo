import numpy as np
import pandas as pd
import pytest

from dropback_analysis.models import (
    clean_term_name,
    fit_logistic_regression,
    majority_class_baseline,
    odds_ratio_table,
    predict_dropback_probability,
)


def make_features(n_per_combo=200):
    # a small synthetic feature table where EMPTY always drops back and I_FORM never does,
    # so the fitted model has an obvious, checkable structure
    rows = []
    for formation, dropback_rate in [('SHOTGUN', 0.5), ('EMPTY', 0.95), ('I_FORM', 0.05)]:
        for i in range(n_per_combo):
            rows.append({
                'offenseFormation': formation,
                'down': 1 + i % 4,
                'yardsToGoBucket': ['short', 'medium', 'long'][i % 3],
                'isDropback': i / n_per_combo < dropback_rate,
            })
    df = pd.DataFrame(rows)
    df['yardsToGoBucket'] = pd.Categorical(
        df['yardsToGoBucket'], categories=['short', 'medium', 'long'], ordered=True
    )
    return df


def test_majority_class_baseline_picks_most_common_class():
    features = pd.DataFrame({'isDropback': [True, True, True, False]})

    result = majority_class_baseline(features)

    assert bool(result['majority_class']) is True
    assert result['baseline_accuracy'] == pytest.approx(0.75)
    assert result['class_shares'].sum() == pytest.approx(1.0)


def test_majority_class_baseline_respects_target_arg():
    features = pd.DataFrame({'handoff': [False, False, False, False, True]})

    result = majority_class_baseline(features, target='handoff')

    assert bool(result['majority_class']) is False
    assert result['baseline_accuracy'] == pytest.approx(0.8)


@pytest.mark.parametrize('raw_name, expected', [
    ("C(offenseFormation, Treatment(reference='SHOTGUN'))[T.EMPTY]", 'offenseFormation: EMPTY'),
    ("C(down, Treatment(reference=1))[T.3]", 'down: 3'),
    ("yardsToGoBucket[T.long]", 'yardsToGoBucket: long'),
    ('Intercept', 'Intercept'),
])
def test_clean_term_name(raw_name, expected):
    assert clean_term_name(raw_name) == expected


def test_fit_logistic_regression_recovers_direction_of_effect():
    features = make_features()

    model = fit_logistic_regression(features)
    odds = np.exp(model.params)

    # EMPTY drops back far more than the SHOTGUN reference, I_FORM far less
    assert odds["C(offenseFormation, Treatment(reference='SHOTGUN'))[T.EMPTY]"] > 1
    assert odds["C(offenseFormation, Treatment(reference='SHOTGUN'))[T.I_FORM]"] < 1


def test_fit_logistic_regression_reference_override_changes_baseline():
    features = make_features()

    model = fit_logistic_regression(features, references={'offenseFormation': 'EMPTY'})

    # with EMPTY as the reference, the other formations become the contrast terms
    terms = ' '.join(model.params.index)
    assert 'T.SHOTGUN' in terms
    assert 'T.EMPTY' not in terms


def test_predict_dropback_probability_matches_a_full_frame_prediction():
    features = make_features()
    model = fit_logistic_regression(features)

    single = predict_dropback_probability(model, down=2, yards_to_go=8, offense_formation='EMPTY')

    # same play pushed through the model as a one-row frame should give the same number
    frame = pd.DataFrame([{'down': 2, 'yardsToGo': 8, 'yardsToGoBucket': 'long', 'offenseFormation': 'EMPTY'}])
    assert single == pytest.approx(float(model.predict(frame).iloc[0]))
    assert 0.0 <= single <= 1.0


def test_predict_dropback_probability_orders_formations_as_expected():
    features = make_features()
    model = fit_logistic_regression(features)

    empty = predict_dropback_probability(model, 3, 10, 'EMPTY')
    shotgun = predict_dropback_probability(model, 3, 10, 'SHOTGUN')
    iform = predict_dropback_probability(model, 3, 10, 'I_FORM')

    # synthetic data: EMPTY nearly always passes, I_FORM nearly never
    assert empty > shotgun > iform


def test_predict_dropback_probability_buckets_yards_to_go():
    features = make_features()
    model = fit_logistic_regression(features)

    # 4 and 6 yards both fall in the 'medium' bucket, so the prediction is identical
    assert predict_dropback_probability(model, 1, 4, 'SHOTGUN') == pytest.approx(
        predict_dropback_probability(model, 1, 6, 'SHOTGUN')
    )


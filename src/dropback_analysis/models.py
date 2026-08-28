import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy.contrasts import Treatment

from dropback_analysis.features import (
    bucket_score_diff,
    bucket_time_left,
    bucket_yards_to_go,
    bucket_yards_to_goal,
)

# reference level for each categorical feature - every coefficient is read relative to this play
DEFAULT_REFERENCES = {
    'offenseFormation': 'SHOTGUN',
    'down': 1,
    'yardsToGoBucket': 'short',
    'scoreDiffBucket': 'tied',
    'yardsToGoalBucket': 'midfield',
    'timeBucket': 'normal',
}


def majority_class_baseline(features, target='isDropback'):
    # the bar any model has to clear: always predict the most common class, use no features
    class_shares = features[target].value_counts(normalize=True)

    return {
        'majority_class': class_shares.idxmax(),
        'baseline_accuracy': class_shares.max(),
        'class_shares': class_shares,
    }


def fit_logistic_regression(features, target='isDropback', references=None):
    # all features treated as categories - the chi-squared step showed their effect isn't linear
    refs = {**DEFAULT_REFERENCES, **(references or {})}
    terms = ' + '.join(f"C({col}, Treatment(reference={ref!r}))" for col, ref in refs.items())
    formula = f"{target}.astype(int) ~ {terms}"

    return smf.logit(formula, data=features).fit()

def predict_dropback_probability(model, down, yards_to_go, offense_formation,
                                 score_diff=0, yards_to_goal=50, seconds_left_in_half=1800):
    # plug in a single play and get P(dropback) back; game-state args default to a neutral situation
    play = pd.DataFrame([{
        'down': down,
        'yardsToGo': yards_to_go,
        'yardsToGoBucket': bucket_yards_to_go(yards_to_go),
        'offenseFormation': offense_formation,
        'scoreDiffBucket': bucket_score_diff(score_diff),
        'yardsToGoalBucket': bucket_yards_to_goal(yards_to_goal),
        'timeBucket': bucket_time_left(seconds_left_in_half),
    }])

    return float(np.asarray(model.predict(play)).ravel()[0])


def clean_term_name(name):
    # "C(offenseFormation, Treatment(reference='SHOTGUN'))[T.EMPTY]" -> "offenseFormation: EMPTY"
    match = re.match(r"C\((\w+).*\)\[T\.(.+)\]", name) or re.match(r"(\w+)\[T\.(.+)\]", name)
    return f"{match.group(1)}: {match.group(2)}" if match else name


def odds_ratio_table(model):
    table = pd.DataFrame({
        'coef': model.params,
        'odds_ratio': np.exp(model.params),
        'p_value': model.pvalues,
    })
    table.index = [clean_term_name(name) for name in table.index]
    table['significant'] = table['p_value'] < 0.05

    return table.sort_values('odds_ratio', ascending=False)

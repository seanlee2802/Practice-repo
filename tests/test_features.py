import pandas as pd
import pytest

from dropback_analysis.features import bucket_yards_to_go, build_features


@pytest.mark.parametrize('yards_to_go, expected_bucket', [
    (1, 'short'),
    (3, 'short'),
    (4, 'medium'),
    (7, 'medium'),
    (8, 'long'),
    (20, 'long'),
])

def test_bucket_yards_to_go(yards_to_go, expected_bucket):
    assert bucket_yards_to_go(yards_to_go) == expected_bucket


def make_plays_df(**overrides):
    row = {
        'gameId': 1,
        'playId': 1,
        'down': 1,
        'yardsToGo': 10,
        'offenseFormation': 'SHOTGUN',
        'isDropback': True,
        'qbKneel': 0,
        'qbSneak': float('nan'),
        'qbSpike': float('nan'),
    }
    row.update(overrides)
    return row


def test_build_features_keeps_normal_decision_plays():
    plays = pd.DataFrame([make_plays_df(playId=1, yardsToGo=10)])

    features = build_features(plays)

    assert len(features) == 1
    assert features.iloc[0]['yardsToGoBucket'] == 'long'


def test_build_features_excludes_kneel_sneak_and_spike_plays():
    plays = pd.DataFrame([
        make_plays_df(playId=1),
        make_plays_df(playId=2, qbKneel=1),
        make_plays_df(playId=3, qbSneak=True),
        make_plays_df(playId=4, qbSpike=True),
    ])

    features = build_features(plays)

    assert sorted(features['playId']) == [1]


def test_build_features_drops_missing_formation():
    plays = pd.DataFrame([
        make_plays_df(playId=1),
        make_plays_df(playId=2, offenseFormation=None),
    ])

    features = build_features(plays)

    assert sorted(features['playId']) == [1]


def test_build_features_output_columns():
    plays = pd.DataFrame([make_plays_df()])

    features = build_features(plays)

    assert list(features.columns) == [
        'gameId', 'playId', 'down', 'yardsToGo', 'offenseFormation', 'isDropback', 'yardsToGoBucket',
    ]
    assert list(features['yardsToGoBucket'].cat.categories) == ['short', 'medium', 'long']

import numpy as np
import pandas as pd


def bucket_yards_to_go(yards_to_go):
    if yards_to_go <= 3:
        return 'short'
    elif yards_to_go <= 7:
        return 'medium'
    else:
        return 'long'


def bucket_score_diff(score_diff):
    # points the offense is ahead (+) or behind (-) at the snap
    if score_diff <= -9:
        return 'trailing_big'
    elif score_diff <= -1:
        return 'trailing'
    elif score_diff == 0:
        return 'tied'
    elif score_diff <= 8:
        return 'leading'
    else:
        return 'leading_big'


def bucket_yards_to_goal(yards_to_goal):
    # distance to the opponent's end zone
    if yards_to_goal <= 20:
        return 'red_zone'
    elif yards_to_goal <= 70:
        return 'midfield'
    else:
        return 'backed_up'


def bucket_time_left(seconds_left_in_half):
    if seconds_left_in_half <= 120:
        return 'two_minute'
    elif seconds_left_in_half <= 300:
        return 'late_half'
    else:
        return 'normal'


def build_features(plays_df, games_df):
    is_kneel_sneak_spike = (
        plays_df[['qbKneel', 'qbSneak', 'qbSpike']]
        .fillna(False)
        .infer_objects(copy=False)
        .astype(bool)
        .any(axis=1)
    )
    decision_plays = plays_df[~is_kneel_sneak_spike]
    decision_plays = decision_plays.dropna(subset=['offenseFormation'])

    d = decision_plays.merge(games_df[['gameId', 'homeTeamAbbr']], on='gameId', how='left')

    # score from the offense's point of view
    home_margin = d['preSnapHomeScore'] - d['preSnapVisitorScore']
    score_diff = np.where(d['possessionTeam'] == d['homeTeamAbbr'], home_margin, -home_margin)

    # yards to the opponent's end zone (a null yardlineSide means the ball is at midfield)
    on_own_side = d['yardlineSide'].fillna(d['possessionTeam']) == d['possessionTeam']
    yards_to_goal = np.where(on_own_side, 100 - d['yardlineNumber'], d['yardlineNumber'])

    # seconds left in the half: Q1/Q3 have another full quarter after them; OT falls through to the clock
    mm_ss = d['gameClock'].fillna('15:00').str.split(':', expand=True).astype(int)
    clock = mm_ss[0] * 60 + mm_ss[1]
    seconds_left_in_half = np.where(d['quarter'].isin([1, 3]), 900, 0) + clock

    features = d[['gameId', 'playId', 'down', 'yardsToGo', 'offenseFormation', 'isDropback']].copy()
    features['yardsToGoBucket'] = features['yardsToGo'].apply(bucket_yards_to_go)
    features['scoreDiffBucket'] = pd.Series(score_diff, index=d.index).apply(bucket_score_diff)
    features['yardsToGoalBucket'] = pd.Series(yards_to_goal, index=d.index).apply(bucket_yards_to_goal)
    features['timeBucket'] = pd.Series(seconds_left_in_half, index=d.index).apply(bucket_time_left)

    features['yardsToGoBucket'] = pd.Categorical(
        features['yardsToGoBucket'], categories=['short', 'medium', 'long'], ordered=True
    )

    return features

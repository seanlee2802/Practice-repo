import matplotlib.pyplot as plt

def plot_play(play_df, game_id, play_id, line_color=None):
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # field background
    ax.set_facecolor('green')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    for yard in range(10, 111, 10):
        ax.axvline(yard, color='white', linewidth=0.5, alpha=0.5)

    # plot each player's path, football separately
    for club, group in play_df.groupby('club'):
        if club == 'football':
            ax.plot(group['x'], group['y'], color='black', label='football')
        else:
            for nfl_id, player_group in group.groupby('nflId'):
                ax.plot(player_group['x'], player_group['y'], color=line_color, alpha=0.7)
            ax.scatter([], [], label=club, color=line_color)

    ax.legend(loc='upper right')
    ax.set_title(f"Game {game_id} - Play {play_id}")
    ax.set_xlabel('x (yards)')
    ax.set_ylabel('y (yards)')
    plt.show()

def get_play(tracking_df, game_id, play_id):
    play_df = tracking_df[
        (tracking_df['gameId'] == game_id) &
        (tracking_df['playId'] == play_id)
    ]
    return game_id, play_id, play_df


def get_random_play(tracking_df):
    game_id, play_id = tracking_df[['gameId', 'playId']].drop_duplicates().sample(1).iloc[0]
    return get_play(tracking_df, game_id, play_id)


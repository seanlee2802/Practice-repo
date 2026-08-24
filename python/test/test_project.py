import numpy as np
import pandas as pd

DATA_DIR = "data"


def main() -> None:
    games = pd.read_csv(f"{DATA_DIR}/games.csv")
    plays = pd.read_csv(f"{DATA_DIR}/plays.csv")

    print(f"Loaded {len(games)} games and {len(plays)} plays")

    yards_to_go = plays["yardsToGo"].to_numpy()
    print(f"Average yards to go: {np.mean(yards_to_go):.2f}")
    print(f"Median yards to go: {np.median(yards_to_go):.2f}")

    top_offenses = plays["possessionTeam"].value_counts().head(5)
    print("\nMost plays run by team:")
    print(top_offenses)


if __name__ == "__main__":
    main()

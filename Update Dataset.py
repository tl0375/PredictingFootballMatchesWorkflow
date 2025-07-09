import os
import requests
import pandas as pd
import glob
from difflib import get_close_matches
from sklearn.preprocessing import LabelEncoder

season_prefix = "2425"

# League codes and names
league_codes = {
    "E0": "Premier League (England)",
    "F1": "Ligue 1 (France)",
    "D1": "Bundesliga (Germany)",
    "I1": "Serie A (Italy)",
    "SP1": "La Liga (Spain)"
}

# Base URL and download folder
base_url = f"https://www.football-data.co.uk/mmz4281/{season_prefix}/"
download_dir = "football_csvs"
os.makedirs(download_dir, exist_ok=True)

# Download CSVs
all_league_dfs = []
for code, name in league_codes.items():
    filename = f"{season_prefix}{code}.csv"
    full_url = base_url + f"{code}.csv"
    save_path = os.path.join(download_dir, filename)

    print(f"Downloading {name} ({code})...")
    response = requests.get(full_url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved: {save_path}")

        # Read and store
        try:
            df = pd.read_csv(save_path)
            df["League"] = code  # Tag with league code
            all_league_dfs.append(df)
        except Exception as e:
            print(f"Could not read {save_path}: {e}")
    else:
        print(f"Failed to download {code}: HTTP {response.status_code}")

# Combine into one DataFrame
if all_league_dfs:
    combined_df = pd.concat(all_league_dfs, ignore_index=True)
    print(f"Combined total rows: {len(combined_df)}")
else:
    print("No valid CSVs downloaded.")
    exit()

# Load existing dataset
dataset_path = "dataset.csv"
if os.path.exists(dataset_path):
    existing_df = pd.read_csv(dataset_path)
else:
    print("No existing dataset.csv found. Creating a new one.")
    existing_df = pd.DataFrame()


# In[38]:


combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%d/%m/%Y', errors='coerce')  
combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')

combined_df = combined_df.iloc[:, :16]
combined_df = combined_df.drop(columns=['Referee'])

combined_df = combined_df.rename(columns={
    'HomeTeam': 'Home_Team',
    'AwayTeam': 'Away_Team',
    'Div': 'League'
})


# In[39]:


name_mapping_newdf = {
    'Man United': 'Manchester Utd',
    'Man City': 'Manchester City',
    'Ath Bilbao' : 'Athletic Club'
}

combined_df['Home_Team'] = combined_df['Home_Team'].replace(name_mapping_newdf)
combined_df['Away_Team'] = combined_df['Away_Team'].replace(name_mapping_newdf)

label_encoder = LabelEncoder()
combined_df['FTR'] = label_encoder.fit_transform(combined_df['FTR'])


# In[41]:


key_fields = ["Date", "Home_Team", "Away_Team"]

# Ensure key columns exist in both DataFrames
for col in key_fields:
    if col not in combined_df.columns or col not in existing_df.columns:
        print(f"Column '{col}' not found in one of the datasets.")
        exit()

combined_key = combined_df[key_fields].astype(str).dropna()
existing_key = existing_df[key_fields].astype(str).dropna()
merged = combined_key.merge(existing_key, on=key_fields, how="left", indicator=True)
new_matches_mask = merged["_merge"] == "left_only"
new_matches_df = combined_df.loc[new_matches_mask.values]

print(f"New matches found: {len(new_matches_df)}")

updated_df = pd.concat([existing_df, new_matches_df], ignore_index=True)


# In[42]:


def compute_gd_elo(df, k=32, initial_rating=1000):
    """
    Compute Goal-Based Elo ratings for each team before each match,
    storing and restoring Elo when a team returns to a league.
    Implements the method by Hvattum & Arntzen (2010).
    """
    teams = {}
    team_leagues = {}
    league_history = {}
    home_elo_list, away_elo_list = [], []

    def goal_multiplier(goal_diff):
        """Adjust K based on goal difference."""
        return 1 + (abs(goal_diff) ** 0.8)  # Scaling function

    for idx, row in df.iterrows():
        home_team, away_team = row['Home_Team'], row['Away_Team']
        home_goals, away_goals = row['FTHG'], row['FTAG']
        goal_diff = home_goals - away_goals  # Goal difference
        home_league, away_league = row['League'], row['League']

        # Initialize Elo if new team
        if home_team not in teams:
            teams[home_team] = initial_rating
        if away_team not in teams:
            teams[away_team] = initial_rating

        home_elo, away_elo = teams[home_team], teams[away_team]

        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)

        # Expected scores
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1.0 - expected_home

        # Actual outcome
        if goal_diff > 0:
            score_home, score_away = 1.0, 0.0
        elif goal_diff < 0:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5

        # K-factor adjustment
        adjusted_k = k * goal_multiplier(goal_diff)

        # Update Elo ratings
        teams[home_team] = home_elo + adjusted_k * (score_home - expected_home)
        teams[away_team] = away_elo + adjusted_k * (score_away - expected_away)

    df['Home_ELO'] = home_elo_list
    df['Away_ELO'] = away_elo_list
    return df


# In[43]:


def compute_streaks(df):
    # Initialize dictionary to store streaks for each team
    team_streaks = {}

    # Lists to store streaks for the dataframe
    home_win_streaks, away_win_streaks = [], []
    home_loss_streaks, away_loss_streaks = [], []
    home_unbeaten_streaks, away_unbeaten_streaks = [], []
    home_winless_streaks, away_winless_streaks = [], []

    # Loop through each match in the dataframe
    for idx, row in df.iterrows():
        home_team, away_team = row['Home_Team'], row['Away_Team']
        ftr = row['FTR']  # FTR = 2 (Home Win), 0 (Away Win), 1 (Draw)

        # Initialize streaks if the team is new
        if home_team not in team_streaks:
            team_streaks[home_team] = {
                'home_win': 0, 'away_win': 0, 'home_loss': 0, 'away_loss': 0,
                'home_unbeaten': 0, 'away_unbeaten': 0, 'home_winless': 0, 'away_winless': 0
            }
        if away_team not in team_streaks:
            team_streaks[away_team] = {
                'home_win': 0, 'away_win': 0, 'home_loss': 0, 'away_loss': 0,
                'home_unbeaten': 0, 'away_unbeaten': 0, 'home_winless': 0, 'away_winless': 0
            }

        # Store current streaks before update
        home_win_streaks.append(team_streaks[home_team]['home_win'])
        away_win_streaks.append(team_streaks[away_team]['away_win'])
        home_loss_streaks.append(team_streaks[home_team]['home_loss'])
        away_loss_streaks.append(team_streaks[away_team]['away_loss'])
        home_unbeaten_streaks.append(team_streaks[home_team]['home_unbeaten'])
        away_unbeaten_streaks.append(team_streaks[away_team]['away_unbeaten'])
        home_winless_streaks.append(team_streaks[home_team]['home_winless'])
        away_winless_streaks.append(team_streaks[away_team]['away_winless'])

        # Update streaks based on result
        if ftr == 2:  # Home Win
            team_streaks[home_team]['home_win'] += 1
            team_streaks[away_team]['away_win'] = 0  # Reset away win streak

            team_streaks[home_team]['home_loss'] = 0  # Reset home loss streak
            team_streaks[away_team]['away_loss'] += 1  # Increase away loss streak

            team_streaks[home_team]['home_unbeaten'] += 1  # Increase home unbeaten streak
            team_streaks[away_team]['away_unbeaten'] = 0  # Reset away unbeaten streak

            team_streaks[home_team]['home_winless'] = 0  # Reset home winless streak
            team_streaks[away_team]['away_winless'] += 1  # Increase away winless streak

        elif ftr == 0:  # Away Win
            team_streaks[away_team]['away_win'] += 1
            team_streaks[home_team]['home_win'] = 0  # Reset home win streak

            team_streaks[away_team]['away_loss'] = 0  # Reset away loss streak
            team_streaks[home_team]['home_loss'] += 1  # Increase home loss streak

            team_streaks[away_team]['away_unbeaten'] += 1  # Increase away unbeaten streak
            team_streaks[home_team]['home_unbeaten'] = 0  # Reset home unbeaten streak

            team_streaks[away_team]['away_winless'] = 0  # Reset away winless streak
            team_streaks[home_team]['home_winless'] += 1  # Increase home winless streak

        else:  # Draw
            team_streaks[home_team]['home_win'] = 0  # Reset home win streak
            team_streaks[away_team]['away_win'] = 0  # Reset away win streak

            team_streaks[home_team]['home_loss'] = 0  # Reset home loss streak
            team_streaks[away_team]['away_loss'] = 0  # Reset away loss streak

            # Increase unbeaten streaks for both teams (separately tracked)
            team_streaks[home_team]['home_unbeaten'] += 1
            team_streaks[away_team]['away_unbeaten'] += 1

            # Increase winless streaks for both teams
            team_streaks[home_team]['home_winless'] += 1
            team_streaks[away_team]['away_winless'] += 1

    # Add calculated streaks to the dataframe
    df['Home_Win_Streak'] = home_win_streaks
    df['Away_Win_Streak'] = away_win_streaks
    df['Home_Loss_Streak'] = home_loss_streaks
    df['Away_Loss_Streak'] = away_loss_streaks
    df['Home_Unbeaten_Streak'] = home_unbeaten_streaks
    df['Away_Unbeaten_Streak'] = away_unbeaten_streaks
    df['Home_Winless_Streak'] = home_winless_streaks
    df['Away_Winless_Streak'] = away_winless_streaks

    return df


# In[44]:


def add_recent_points(df, window_sizes=[5, 10, 20]):
    """
    Adds rolling points over the last 'n' games for home and away teams.

    Parameters:
    df (pd.DataFrame): The match dataset with teams and results.
    window_sizes (list): The number of past games to consider (default: [5, 10]).

    Returns:
    pd.DataFrame: Updated dataframe with rolling points for each team.
    """
    team_points = {}  # Dictionary to track team results over time

    # Initialize columns for each rolling window
    for window in window_sizes:
        df[f"Home_Points_Last_{window}"] = 0
        df[f"Away_Points_Last_{window}"] = 0

    # Iterate over matches
    for idx, row in df.iterrows():
        home_team, away_team = row["Home_Team"], row["Away_Team"]
        ftr = row["FTR"]  # 2 = Home Win, 1 = Draw, 0 = Away Win

        # Assign points based on result
        home_points, away_points = 0, 0
        if ftr == 2:  # Home win
            home_points, away_points = 3, 0
        elif ftr == 1:  # Draw
            home_points, away_points = 1, 1
        elif ftr == 0:  # Away win
            home_points, away_points = 0, 3

        # Initialize history for teams if not present
        if home_team not in team_points:
            team_points[home_team] = []
        if away_team not in team_points:
            team_points[away_team] = []

        # Store rolling points before updating
        for window in window_sizes:
            df.at[idx, f"Home_Points_Last_{window}"] = sum(team_points[home_team][-window:])
            df.at[idx, f"Away_Points_Last_{window}"] = sum(team_points[away_team][-window:])

        # Append the latest match result to the team's history
        team_points[home_team].append(home_points)
        team_points[away_team].append(away_points)

    return df


# In[45]:


def add_net_stats(df):
    stat_pairs = {
        "FTG": ("FTHG", "FTAG"),
        "S": ("HS", "AS"),
        "ST": ("HST", "AST"),
    }
    # Loop through stat pairs and calculate net stats
    for stat_name, (home_stat, away_stat) in stat_pairs.items():
        df[f"Home_Net_{stat_name}"] = df[home_stat] - df[away_stat]
        df[f"Away_Net_{stat_name}"] = df[away_stat] - df[home_stat]

    return df


# In[46]:


def add_conceded_stats(df):
    df["FTHG_Conceded"] = df["FTAG"]
    df["FTAG_Conceded"] = df["FTHG"]

    df["HST_Conceded"] = df["AST"]
    df["AST_Conceded"] = df["HST"]
    
    df["HS_Conceded"] = df["AS"]
    df["AS_Conceded"] = df["HS"]
    
    return df


# In[47]:


def calculate_recent_performance(df, num_games):
    """
    Calculate rolling sums/means for points, xG, goals, and Net statistics.
    """
    df['Home_Rolling_Goals_For'] = df.groupby('Home_Team')['FTHG'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )
    df['Away_Rolling_Goals_For'] = df.groupby('Away_Team')['FTAG'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )

    df['Home_Rolling_Goals_Against'] = df.groupby('Home_Team')['FTAG'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )
    df['Away_Rolling_Goals_Against'] = df.groupby('Away_Team')['FTHG'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )

    df['Home_Rolling_Net_FTG'] = df.groupby('Home_Team')['Home_Net_FTG'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )
    df['Away_Rolling_Net_FTG'] = df.groupby('Away_Team')['Away_Net_FTG'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )

    df['Home_Rolling_Net_ST'] = df.groupby('Home_Team')['Home_Net_ST'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )
    df['Away_Rolling_Net_ST'] = df.groupby('Away_Team')['Away_Net_ST'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )

    df['Home_Rolling_Net_S'] = df.groupby('Home_Team')['Home_Net_S'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )
    df['Away_Rolling_Net_S'] = df.groupby('Away_Team')['Away_Net_S'].transform(
        lambda x: x.rolling(num_games).mean().shift()
    )
    
    return df


# In[48]:


updated_df = compute_gd_elo(updated_df)
updated_df = compute_streaks(updated_df)
updated_df = add_recent_points(updated_df)


# In[49]:


updated_df = add_net_stats(updated_df)
updated_df = add_conceded_stats(updated_df)
updated_df = calculate_recent_performance(updated_df, num_games=10)


# In[50]:


updated_df.to_csv(dataset_path, index=False)
print(f"dataset.csv updated: {len(updated_df)} total rows")


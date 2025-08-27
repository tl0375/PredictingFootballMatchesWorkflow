import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from xgboost import XGBClassifier
from datetime import datetime
from zoneinfo import ZoneInfo

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DualLSTMWithEloXGBoost(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_layers, mlp_input_size, mlp_hidden_size, xgb_input_size, num_classes, dropout):
        super(DualLSTMWithEloXGBoost, self).__init__()
        self.home_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                                 num_layers=lstm_layers, batch_first=True,
                                 dropout=dropout if lstm_layers > 1 else 0.0)
        self.away_lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                                 num_layers=lstm_layers, batch_first=True,
                                 dropout=dropout if lstm_layers > 1 else 0.0)
        self.layer_norm_home = nn.LayerNorm(lstm_hidden_size)
        self.layer_norm_away = nn.LayerNorm(lstm_hidden_size)
        self.elo_mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.final_layer = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size + (mlp_hidden_size // 2) + xgb_input_size,
                      (2 * lstm_hidden_size + (mlp_hidden_size // 2) + xgb_input_size) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((2 * lstm_hidden_size + (mlp_hidden_size // 2) + xgb_input_size) // 2, num_classes)
        )
    def forward(self, home_seq, away_seq, elo, xgb):
        home_out, _ = self.home_lstm(home_seq)
        away_out, _ = self.away_lstm(away_seq)
        home_last = home_out[:, -1, :]
        away_last = away_out[:, -1, :]
        home_last = self.layer_norm_home(home_last)
        away_last = self.layer_norm_away(away_last)
        elo_out = self.elo_mlp(elo)
        combined = torch.cat([home_last, away_last, elo_out, xgb], dim=1)
        return self.final_layer(combined)

# Functions reused from training script:
def get_team_matches_before_date(df, team, match_date, n):
    df_team = df[((df['Home_Team'] == team) | (df['Away_Team'] == team)) & (df['Date'] < match_date)]
    df_team = df_team.sort_values(by='Date').tail(n)
    return df_team

def pad_sequence(arr, seq_len, num_feats):
    current_len = arr.shape[0] if arr.size > 0 else 0
    if current_len == seq_len:
        return arr
    elif current_len > seq_len:
        return arr[-seq_len:]
    else:
        padding = np.zeros((seq_len - current_len, num_feats), dtype=arr.dtype)
        return np.concatenate([padding, arr], axis=0)

def build_single_sequence(df, seq_len, lstm_features, elo_features, xgb_features, home_team, away_team, match_date):
    # Build sequences & features for one fixture
    df_sorted = df.sort_values(by='Date').reset_index(drop=True)

    # Get recent matches for home and away teams
    df_home = get_team_matches_before_date(df_sorted, home_team, match_date, seq_len)
    df_away = get_team_matches_before_date(df_sorted, away_team, match_date, seq_len)

    # Handle missing sequence data using 10th percentile fallback
    if len(df_home) == 0:
        q25_home = df[lstm_features].quantile(0.1).values.astype(np.float32)
        home_features = np.tile(q25_home, (seq_len, 1))
        print('data missing for', home_team, away_team)
    else:
        home_features = pad_sequence(df_home[lstm_features].values, seq_len, len(lstm_features))

    if len(df_away) == 0:
        q25_away = df[lstm_features].quantile(0.1).values.astype(np.float32)
        away_features = np.tile(q25_away, (seq_len, 1))
    else:
        away_features = pad_sequence(df_away[lstm_features].values, seq_len, len(lstm_features))

    # For Elo and XGB features, try exact match row first
    df_fixture = df_sorted[
        (df_sorted['Home_Team'] == home_team) &
        (df_sorted['Away_Team'] == away_team) &
        (df_sorted['Date'] == match_date)
    ]

    if len(df_fixture) == 0:
        # If no exact match, get most recent match between the two before the match_date
        df_fixture = df_sorted[
            (df_sorted['Home_Team'] == home_team) &
            (df_sorted['Away_Team'] == away_team) &
            (df_sorted['Date'] < match_date)
        ].tail(1)

        if len(df_fixture) == 0:
            # Fallback to 10th percentile if no fixture info at all
            elo = df[elo_features].quantile(0.1).values.astype(np.float32)
            xgb = df[xgb_features].quantile(0.1).values.astype(np.float32)
        else:
            elo = df_fixture[elo_features].values[0].astype(np.float32)
            xgb = df_fixture[xgb_features].values[0].astype(np.float32)
    else:
        elo = df_fixture[elo_features].values[0].astype(np.float32)
        xgb = df_fixture[xgb_features].values[0].astype(np.float32)

    return home_features, away_features, elo, xgb


def predict_fixture(model, xgb_model, scaler_lstm, scaler_elo, scaler_xgb, df, seq_len,
                    lstm_features, elo_features, xgb_features,
                    home_team, away_team, match_date_str,
                    home_promoted=0, away_promoted=0): 
    match_date = pd.to_datetime(match_date_str)
    model.eval()
    
    # Build feature sequences for this fixture
    home_seq_np, away_seq_np, elo_np, xgb_np = build_single_sequence(df, seq_len, lstm_features, elo_features, xgb_features,
                                                                     home_team, away_team, match_date)
    # Scale LSTM features (reshape for scaler)
    home_seq_scaled = scaler_lstm.transform(home_seq_np.reshape(-1, len(lstm_features))).reshape(seq_len, len(lstm_features))
    away_seq_scaled = scaler_lstm.transform(away_seq_np.reshape(-1, len(lstm_features))).reshape(seq_len, len(lstm_features))
    elo_scaled = scaler_elo.transform(elo_np.reshape(1, -1)).reshape(-1)
    xgb_scaled = scaler_xgb.transform(xgb_np.reshape(1, -1)).reshape(-1)

    # XGBoost prediction for XGB features (must produce proba vector)
    xgb_pred_proba = xgb_model.predict_proba(xgb_scaled.reshape(1, -1))[0]

    # Convert to torch tensors and add batch dim
    home_seq_t = torch.tensor(home_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    away_seq_t = torch.tensor(away_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    elo_t = torch.tensor(elo_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    xgb_t = torch.tensor(xgb_pred_proba, dtype=torch.float32).unsqueeze(0).to(device)

    # Forward pass through model
    with torch.no_grad():
        output = model(home_seq_t, away_seq_t, elo_t, xgb_t)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    classes = ['Away Win', 'Draw', 'Home Win']
    pred_class = classes[np.argmax(probs)]

    return pred_class, probs

def adjust_probs_for_promotion(probs, home_promoted, away_promoted):
    """
    probs: np.array or list-like of [away_win_prob, draw_prob, home_win_prob]
    home_promoted, away_promoted: booleans or 0/1
    
    Returns adjusted probs (np.array), still summing to 1.
    Properly adjusts probabilties fpr newly promoted teams who may not have any existing data
    """
    probs = np.array(probs, dtype=float)  # Work on a copy

    adjustment_factor = 0.7  # reduce winning chance by 30% 

    if home_promoted:
        original_home_win = probs[2]
        probs[2] *= adjustment_factor  # reduce home win prob
        leftover = original_home_win - probs[2]  # leftover from reduction
        probs[0] += leftover  # add to away win

    if away_promoted:
        original_away_win = probs[0]
        probs[0] *= adjustment_factor  # reduce away win prob
        leftover = original_away_win - probs[0]  # leftover from reduction
        probs[2] += leftover    # add to home win

    # Normalize to sum to 1
    probs /= probs.sum()
    return probs


if __name__ == "__main__":
    import sys

    # File paths
    model_path = "Testing/best_model.pth"
    xgb_path = "Testing/xgb_model.json"
    scaler_lstm_path = "Testing/scaler_lstm.pkl"
    scaler_elo_path = "Testing/scaler_elo.pkl"
    scaler_xgb_path = "Testing/scaler_xgb.pkl"
    data_csv = "dataset.csv"      # Historical match data with all features
    
    fixtures_csv = "fixtures_clean.csv" # New CSV containing fixtures to predict

    # Feature lists, matching training
    lstm_features = [
        'FTHG', 'FTAG', 'HS', 'HST', 'AS', 'AST',
        "Home_Net_FTG", "Away_Net_FTG",
        "Home_Net_ST", "Away_Net_ST",
        "Home_Net_S", "Away_Net_S",
        "FTHG_Conceded", "FTAG_Conceded",
        "HST_Conceded", "AST_Conceded",
        "HS_Conceded", "AS_Conceded"
    ]
    elo_features = ['Home_ELO', 'Away_ELO', 'Home_Newly_Promoted', 'Away_Newly_Promoted']
    xgb_features = ['Home_Win_Streak', 'Away_Win_Streak', 'Home_Loss_Streak', 'Away_Loss_Streak', 
                    'Home_Unbeaten_Streak', 'Away_Unbeaten_Streak', 'Home_Winless_Streak', 'Away_Winless_Streak',
                    'Home_Points_Last_5', 'Home_Points_Last_10', 'Home_Points_Last_20', 
                    'Away_Points_Last_5', 'Away_Points_Last_10', 'Away_Points_Last_20'
                   ]

    # Load scalers
    scaler_lstm = joblib.load(scaler_lstm_path)
    scaler_elo = joblib.load(scaler_elo_path)
    scaler_xgb = joblib.load(scaler_xgb_path)

    # Load XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.load_model(xgb_path)

    # Load PyTorch model
    model = DualLSTMWithEloXGBoost(
        lstm_input_size=len(lstm_features),
        lstm_hidden_size=64,
        lstm_layers=5,
        mlp_input_size=len(elo_features),
        mlp_hidden_size=16,
        xgb_input_size=3,
        num_classes=3,
        dropout=0.2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load historical match data with all features
    df = pd.read_csv(data_csv)
    df['Date'] = pd.to_datetime(df['Date'])

    # Load fixtures to predict
    fixtures = pd.read_csv(fixtures_csv)
    fixtures['Date'] = pd.to_datetime(fixtures['Date'])

    #Define newly promoted teams
    newly_promoted_teams = {
        'Sunderland', 'Leeds', 'Burnley',
        'Levante', 'Elche', 'Real Oviedo',
        'Pisa', 'Cremonese', 'Sassuolo',
        'FC Koln', 'Hamburger SV',
        'Lorient', 'Paris FC', 'Metz'
    }
    
    # Flag newly promoted teams in fixtures DataFrame
    fixtures['Home_Newly_Promoted'] = fixtures['Home_Team'].isin(newly_promoted_teams).astype(int)
    fixtures['Away_Newly_Promoted'] = fixtures['Away_Team'].isin(newly_promoted_teams).astype(int)

    results = []

    for idx, row in fixtures.iterrows():
        home_team = row['Home_Team']
        away_team = row['Away_Team']
        match_date_str = row['Date'].strftime('%Y-%m-%d')
        home_promoted = int(home_team in newly_promoted_teams)
        away_promoted = int(away_team in newly_promoted_teams)

        if idx == 0:
            match_date = pd.to_datetime(match_date_str)
            df_home = get_team_matches_before_date(df, home_team, match_date, 10)
            df_away = get_team_matches_before_date(df, away_team, match_date, 10)
    
            print(f"\n[DEBUG] Fixture: {home_team} vs {away_team} on {match_date_str}")
            print("[DEBUG] Home last 10 matches:")
            print(df_home[['Date', 'Home_Team', 'Away_Team', 'FTHG', 'FTAG']])
            print("[DEBUG] Away last 10 matches:")
            print(df_away[['Date', 'Home_Team', 'Away_Team', 'FTHG', 'FTAG']])
            

        try:
            pred_class, probs = predict_fixture(
                model, xgb_model, scaler_lstm, scaler_elo, scaler_xgb,
                df, seq_len=10,
                lstm_features=lstm_features,
                elo_features=elo_features,
                xgb_features=xgb_features,
                home_team=home_team,
                away_team=away_team,
                match_date_str=match_date_str,
                home_promoted=home_promoted,   
                away_promoted=away_promoted    
            )

            probs = adjust_probs_for_promotion(probs, home_promoted, away_promoted)
            # Save prediction in a structured way
            results.append({
                'Date': match_date_str,
                'Time': row['Time'],
                'Matchweek': row['Matchweek'],
                'League': row['Competition'],
                'Home_Team': home_team,
                'Away_Team': away_team,
                'Predicted_Result': pred_class,
                'Prob_Home_Win': probs[2],
                'Prob_Draw': probs[1],
                'Prob_Away_Win': probs[0]
            })

        except Exception as e:
            print(f"Error predicting fixture {home_team} vs {away_team} on {match_date_str}: {e}")
            results.append({
                'Date': match_date_str,
                'Time': row['Time'],
                'Matchweek': row['Matchweek'],
                'League': row['Competition'],
                'Home_Team': home_team,
                'Away_Team': away_team,
                'Predicted_Result': 'Error',
                'Prob_Home_Win': np.nan,
                'Prob_Draw': np.nan,
                'Prob_Away_Win': np.nan
            })

    # Save all predictions to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("football-predictor-ui/Results.csv", index=False)
    print("Saved all predictions to Results.csv")

uk_time = datetime.now(ZoneInfo("Europe/London"))
formatted_time = uk_time.strftime("%Y-%m-%d %H:%M:%S")

with open("football-predictor-ui/last_updated.txt", "w") as f:
    f.write(formatted_time)


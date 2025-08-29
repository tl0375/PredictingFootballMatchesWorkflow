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
        home_last = home_last 
        away_last = away_last
        combined = torch.cat([home_last, away_last, elo_out, xgb], dim=1)
        return self.final_layer(combined)

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

def build_single_sequence(df, seq_len, lstm_features, elo_features, xgb_features,
                          home_team, away_team, match_date):
    df_sorted = df.sort_values(by='Date').reset_index(drop=True)

    # --- LSTM sequences ---
    df_home_seq = get_team_matches_before_date(df_sorted, home_team, match_date, seq_len)
    df_away_seq = get_team_matches_before_date(df_sorted, away_team, match_date, seq_len)


    if len(df_home_seq) == 0:
        home_features = np.tile(df[lstm_features].quantile(0.1).values.astype(np.float32), (seq_len, 1))
    else:
        home_features = pad_sequence(df_home_seq[lstm_features].values, seq_len, len(lstm_features))

    if len(df_away_seq) == 0:
        away_features = np.tile(df[lstm_features].quantile(0.1).values.astype(np.float32), (seq_len, 1))
    else:
        away_features = pad_sequence(df_away_seq[lstm_features].values, seq_len, len(lstm_features))

    
    elo = np.zeros(4, dtype=np.float32)
    xgb = np.zeros(len(xgb_features), dtype=np.float32)
    
    # --- Helper: copy streak/points stats ---
    def map_features(row, team_role_last, team_role_fixture):
        vals = {}
        for feat in xgb_features:
            base = feat[4:]  # strip Home_/Away_
            last_col = f"{team_role_last}{base}"
            fixture_col = f"{team_role_fixture}{base}"
    
            if last_col in row.index and fixture_col in xgb_features:
                vals[fixture_col] = row[last_col]
        return vals

        # --- Home team ---
    if len(df_home_seq) > 0:
        home_row = df_home_seq.iloc[-1]
        if home_row['Home_Team'] == home_team:
            elo[0] = home_row['Home_ELO']
            elo[2] = 0 if pd.isna(home_row['Home_Newly_Promoted']) else home_row['Home_Newly_Promoted']
            mapped = map_features(home_row, "Home", "Home")
        else:
            elo[0] = home_row['Away_ELO']
            elo[2] = 0 if pd.isna(home_row['Away_Newly_Promoted']) else home_row['Away_Newly_Promoted']
            mapped = map_features(home_row, "Away", "Home")
    
        for feat, val in mapped.items():
            xgb[xgb_features.index(feat)] = val
    else:
        # fallback if no history
        elo[:2] = df[elo_features].quantile(0.1).values[:2]
        elo[2] = 0  # promoted flag should always default to 0
        mapped = {feat: df[xgb_features].quantile(0.1)[feat] for feat in xgb_features if feat.startswith("Home")}
        for feat, val in mapped.items():
            xgb[xgb_features.index(feat)] = val
    
    # --- Away team ---
    if len(df_away_seq) > 0:
        away_row = df_away_seq.iloc[-1]
        if away_row['Home_Team'] == away_team:
            elo[1] = away_row['Home_ELO']
            elo[3] = 0 if pd.isna(away_row['Home_Newly_Promoted']) else away_row['Home_Newly_Promoted']
            mapped = map_features(away_row, "Home", "Away")
        else:
            elo[1] = away_row['Away_ELO']
            elo[3] = 0 if pd.isna(away_row['Away_Newly_Promoted']) else away_row['Away_Newly_Promoted']
            mapped = map_features(away_row, "Away", "Away")
    
        for feat, val in mapped.items():
            xgb[xgb_features.index(feat)] = val
    else:
        elo[1] = df[elo_features].quantile(0.1).values[1]  # away elo
        elo[3] = 0  # promoted flag should always default to 0
        mapped = {feat: df[xgb_features].quantile(0.1)[feat] for feat in xgb_features if feat.startswith("Away")}
        for feat, val in mapped.items():
            xgb[xgb_features.index(feat)] = val

    return home_features, away_features, elo, xgb


def predict_fixture(model, xgb_model, scaler_lstm, scaler_elo, scaler_xgb, df, seq_len,
                    lstm_features, elo_features, xgb_features,
                    home_team, away_team, match_date_str,
                    home_promoted=0, away_promoted=0,
                    elo_sensitivity=2, xgb_sensitivity=2, temperature=1.4,
                    first_fixture=False):  # flag to run sensitivity checks if needing superficial probability tweaks
    match_date = pd.to_datetime(match_date_str)
    model.eval()
    
    # Build feature sequences for this fixture
    home_seq_np, away_seq_np, elo_np, xgb_np = build_single_sequence(
        df, seq_len, lstm_features, elo_features, xgb_features,
        home_team, away_team, match_date
    )

    if home_seq_np.size == 0:
        raise ValueError(f"No LSTM features for {home_team} before {match_date_str}")
    if away_seq_np.size == 0:
        raise ValueError(f"No LSTM features for {away_team} before {match_date_str}")

    # Scale LSTM features
    home_seq_scaled = scaler_lstm.transform(
        home_seq_np.reshape(-1, len(lstm_features))
    ).reshape(seq_len, len(lstm_features))

    away_seq_scaled = scaler_lstm.transform(
        away_seq_np.reshape(-1, len(lstm_features))
    ).reshape(seq_len, len(lstm_features))


    # Scale ELO features 
    elo_scaled = scaler_elo.transform(elo_np.reshape(1, -1)).reshape(-1)
    
    # Scale XGB features 
    xgb_scaled = scaler_xgb.transform(xgb_np.reshape(1, -1)).reshape(-1)

    elo_scaled = elo_scaled * elo_sensitivity + (1 - elo_sensitivity) * 0
    xgb_scaled = xgb_scaled * xgb_sensitivity + (1 - xgb_sensitivity) * 0
    
    # XGBoost prediction (logit scaling to make model more sensitive to changes)
    xgb_pred_proba = xgb_model.predict_proba(xgb_scaled.reshape(1, -1))[0]

    # Convert to torch tensors
    home_seq_t = torch.tensor(home_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    away_seq_t = torch.tensor(away_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    elo_t = torch.tensor(elo_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    xgb_t = torch.tensor(xgb_pred_proba, dtype=torch.float32).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(home_seq_t, away_seq_t, elo_t, xgb_t)
        probs = torch.softmax(output / temperature, dim=1).cpu().numpy()[0]
        # Adjust home probability downward by a factor
        home_bias_factor = 0.8  # reduce probability of home wins by 20%
        probs[2] *= home_bias_factor
        draw_bias_factor = 0.7 # reduce probability of draws by 30%
        probs[1] *= draw_bias_factor
        # Re-normalize so all sum to 1
        probs /= probs.sum()

        #optional sensitivity test, triggered by first_fixture flag
        if first_fixture:
            with torch.no_grad():
                # Baseline logits and probs
                output = model(home_seq_t, away_seq_t, elo_t, xgb_t)
                baseline_probs = torch.softmax(output / temperature, dim=1).cpu().numpy()[0]
                print(f"Baseline probs: {baseline_probs}")
                test_sensitivities = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
                for e_mult in test_sensitivities:
                    for x_mult in test_sensitivities:
                        if e_mult == 1.0 and x_mult == 1.0:
                            continue
                            
                        output_test = model(
                            home_seq_t,
                            away_seq_t,
                            elo_t * e_mult,
                            xgb_t * x_mult
                        )

                        probs_test = torch.softmax(output_test / temperature, dim=1).cpu().numpy()[0]
                        delta = probs_test - baseline_probs
                        print(f"ELO={e_mult}, XGB={x_mult} → probs={probs_test}, Δ={delta}")


    classes = ['Away Win', 'Draw', 'Home Win']
    pred_class = classes[np.argmax(probs)]

    return pred_class, probs

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

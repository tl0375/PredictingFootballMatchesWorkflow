import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset    #Import neccessary libraries and modules
import torch.nn.utils as utils
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Define device (automatically uses CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  

def get_team_matches_before_date(df, team, match_date, n):
    """
    Retrieves the last n matches for a given team before a specified date.
    
    Args:
        df (pd.DataFrame): Complete match dataset
        team (str): Team name to filter matches
        match_date (datetime): Cutoff date for matches
        n (int): Number of matches to retrieve
        
    Returns:
        pd.DataFrame: Filtered dataframe with team's last n matches before match_date
    """
    df_team = df[((df['Home_Team'] == team) | (df['Away_Team'] == team)) & (df['Date'] < match_date)]
    df_team = df_team.sort_values(by='Date').tail(n)
    return df_team

def pad_sequence(arr, seq_len, num_feats):
    """
    Pads or truncates a sequence to a fixed length.
    
    Args:
        arr (np.array): Input sequence array
        seq_len (int): Target sequence length
        num_feats (int): Number of features per timestep
        
    Returns:
        np.array: Padded/truncated array of shape (seq_len, num_feats)
    """
    current_len = arr.shape[0] if arr.size > 0 else 0
    if current_len == seq_len:
        return arr
    elif current_len > seq_len:
        return arr[-seq_len:]
    else:
        padding = np.zeros((seq_len - current_len, num_feats), dtype=arr.dtype)
        return np.concatenate([padding, arr], axis=0)

def build_sequences(df, seq_len, lstm_features, elo_features, xgb_features, target_col='FTR'):
    """
    Constructs input sequences for the hybrid model.
    
    Args:
        df (pd.DataFrame): Raw match data
        seq_len (int): Number of historical matches to consider
        lstm_features (list): Features for LSTM sequences
        elo_features (list): Elo rating features
        xgb_features (list): Features for XGBoost model
        target_col (str): Target variable column name
        
    Returns:
        tuple: (home_sequences, away_sequences, elo_inputs, xgb_inputs, labels)
    """
    df_sorted = df.sort_values(by='Date').reset_index(drop=True)
    
    home_seqs, away_seqs, elo_inputs, xgb_inputs, labels = [], [], [], [], []

    for _, row in df_sorted.iterrows():
        home_team, away_team, match_date = row['Home_Team'], row['Away_Team'], row['Date']

        # Get past matches for each team
        df_home = get_team_matches_before_date(df_sorted, home_team, match_date, seq_len)
        df_away = get_team_matches_before_date(df_sorted, away_team, match_date, seq_len)

        # Extract features for LSTM (including FTR from previous matches)
        home_features = pad_sequence(df_home[lstm_features].values if len(df_home) > 0 else np.zeros((0, len(lstm_features))), seq_len, len(lstm_features))
        away_features = pad_sequence(df_away[lstm_features].values if len(df_away) > 0 else np.zeros((0, len(lstm_features))), seq_len, len(lstm_features))

        # Elo ratings as additional features
        elo = row[elo_features].values.astype(np.float32)

        # XGBoost features 
        xgb = row[xgb_features].values.astype(np.float32)

        home_seqs.append(home_features)
        away_seqs.append(away_features)
        elo_inputs.append(elo)
        xgb_inputs.append(xgb)
        labels.append(row[target_col])  

    return np.array(home_seqs), np.array(away_seqs), np.array(elo_inputs), np.array(xgb_inputs), np.array(labels)

class DualLSTMWithEloXGBoost(nn.Module):
    """
    Hybrid neural network combining:
    - Dual LSTMs for home/away team sequence processing
    - MLP for Elo ratings
    - XGBoost feature integration
    
    Architecture:
    1. Two separate LSTMs process home and away team sequences
    2. MLP processes current Elo ratings
    3. XGBoost predictions are incorporated
    4. All components are concatenated and passed through final full connected dense layers
    
    Args:
        lstm_input_size (int): Number of features per timestep
        lstm_hidden_size (int): LSTM hidden state size
        lstm_layers (int): Number of LSTM layers
        mlp_input_size (int): Number of Elo features
        mlp_hidden_size (int): MLP hidden layer size
        xgb_input_size (int): Number of XGBoost features
        num_classes (int): Number of output classes (3 for football: Win, Draw and Loss)
        dropout (float): Dropout probability
    """
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_layers, mlp_input_size, mlp_hidden_size, xgb_input_size, num_classes, dropout):
        super(DualLSTMWithEloXGBoost, self).__init__()
        
        self.home_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,  
            dropout=dropout if lstm_layers > 1 else 0.0  
        )
        self.away_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

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
            nn.Linear(2 * lstm_hidden_size + (mlp_hidden_size // 2) + xgb_input_size, (2 * lstm_hidden_size + (mlp_hidden_size // 2) + xgb_input_size) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((2 * lstm_hidden_size + (mlp_hidden_size // 2) + xgb_input_size) // 2, num_classes)
        )

    def forward(self, home_seq, away_seq, elo, xgb):
        home_out, _ = self.home_lstm(home_seq)
        away_out, _ = self.away_lstm(away_seq)
        
        # Take the output from the last time step
        home_last = home_out[:, -1, :]
        away_last = away_out[:, -1, :]
    
        home_last = self.layer_norm_home(home_last)
        away_last = self.layer_norm_away(away_last)
        
        elo_out = self.elo_mlp(elo)
    
        # Combine LSTM outputs, Elo MLP output, and XGBoost features
        combined = torch.cat([home_last, away_last, elo_out, xgb], dim=1)
        return self.final_layer(combined)

#Load dataset from .csv file
df = pd.read_csv("dataset.csv")


# Feature selection for each part of the model
lstm_features = [
    'FTHG', 'FTAG', 'HS', 'HST', 'AS', 'AST',
    "Home_Net_FTG", "Away_Net_FTG",
    "Home_Net_ST", "Away_Net_ST",
    "Home_Net_S", "Away_Net_S",
    "FTHG_Conceded", "FTAG_Conceded",
    "HST_Conceded", "AST_Conceded",
    "HS_Conceded", "AS_Conceded"
]
elo_features = ['Home_ELO', 'Away_ELO']
xgb_features = ['Home_Win_Streak', 'Away_Win_Streak', 'Home_Loss_Streak', 'Away_Loss_Streak', 
                'Home_Unbeaten_Streak', 'Away_Unbeaten_Streak', 'Home_Winless_Streak', 'Away_Winless_Streak',
                'Home_Points_Last_5', 'Home_Points_Last_10', 'Home_Points_Last_20', 
                'Away_Points_Last_5', 'Away_Points_Last_10', 'Away_Points_Last_20'
               ]

# Define fixed input sizes
mlp_input_size = len(elo_features)  
xgb_input_size = len(xgb_features)  

seq_len = 10  #Number of matches backward to get stats from
X_home, X_away, X_elo, X_xgb, y = build_sequences(df, seq_len, lstm_features, elo_features, xgb_features, 'FTR')

# Split data chronologically by league (70% train, 20% validation, 10% test)
df_train_list, df_valid_list, df_test_list = [], [], []
for league, group in df.groupby("League"):
    group_sorted = group.sort_values("Date")
    n = len(group_sorted)
    train_end = int(0.7 * n)
    valid_end = int(0.9 * n)

    df_train_list.append(group_sorted.iloc[:train_end])
    df_valid_list.append(group_sorted.iloc[train_end:valid_end])
    df_test_list.append(group_sorted.iloc[valid_end:])

df_train = pd.concat(df_train_list).reset_index(drop=True)
df_valid = pd.concat(df_valid_list).reset_index(drop=True)
df_test = pd.concat(df_test_list).reset_index(drop=True)

X_home_train, X_away_train, X_elo_train, X_xgb_train, y_train = build_sequences(df_train, seq_len, lstm_features, elo_features, xgb_features, 'FTR')
X_home_valid, X_away_valid, X_elo_valid, X_xgb_valid, y_valid = build_sequences(df_valid, seq_len, lstm_features, elo_features, xgb_features, 'FTR')
X_home_test, X_away_test, X_elo_test, X_xgb_test, y_test = build_sequences(df_test, seq_len, lstm_features, elo_features, xgb_features, 'FTR')

# Standard scaling is applied separately to:
# 1. LSTM sequence features
# 2. Elo ratings
# 3. XGBoost features

scaler_lstm = StandardScaler()
N, T, F = X_home_train.shape
X_home_train = scaler_lstm.fit_transform(X_home_train.reshape(-1, F)).reshape(N, T, F)
X_away_train = scaler_lstm.transform(X_away_train.reshape(-1, F)).reshape(N, T, F)

X_home_valid = scaler_lstm.transform(X_home_valid.reshape(-1, F)).reshape(X_home_valid.shape[0], T, F)
X_away_valid = scaler_lstm.transform(X_away_valid.reshape(-1, F)).reshape(X_away_valid.shape[0], T, F)
X_home_test = scaler_lstm.transform(X_home_test.reshape(-1, F)).reshape(X_home_test.shape[0], T, F)
X_away_test = scaler_lstm.transform(X_away_test.reshape(-1, F)).reshape(X_away_test.shape[0], T, F)

scaler_elo = StandardScaler()
X_elo_train = scaler_elo.fit_transform(X_elo_train)
X_elo_valid = scaler_elo.transform(X_elo_valid)
X_elo_test = scaler_elo.transform(X_elo_test)


scaler_xgb = StandardScaler()
X_xgb_train = scaler_xgb.fit_transform(X_xgb_train)
X_xgb_valid = scaler_xgb.transform(X_xgb_valid)
X_xgb_test = scaler_xgb.transform(X_xgb_test)

#XGBoost model is defined with parameters
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=0.1,
    min_child_weight=3,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_xgb_train, y_train)

# Generate XGBoost predictions
X_xgb_train_pred = xgb_model.predict_proba(X_xgb_train)  
X_xgb_valid_pred = xgb_model.predict_proba(X_xgb_valid)
X_xgb_test_pred = xgb_model.predict_proba(X_xgb_test)

# Convert to tensors
X_home_train_t = torch.tensor(X_home_train, dtype=torch.float32)
X_away_train_t = torch.tensor(X_away_train, dtype=torch.float32)
X_elo_train_t = torch.tensor(X_elo_train, dtype=torch.float32)
X_xgb_train_t = torch.tensor(X_xgb_train_pred, dtype=torch.float32)  
y_train_t = torch.tensor(y_train, dtype=torch.long)

X_home_valid_t = torch.tensor(X_home_valid, dtype=torch.float32)
X_away_valid_t = torch.tensor(X_away_valid, dtype=torch.float32)
X_elo_valid_t = torch.tensor(X_elo_valid, dtype=torch.float32)
X_xgb_valid_t = torch.tensor(X_xgb_valid_pred, dtype=torch.float32)  
y_valid_t = torch.tensor(y_valid, dtype=torch.long)

X_home_test_t = torch.tensor(X_home_test, dtype=torch.float32)
X_away_test_t = torch.tensor(X_away_test, dtype=torch.float32)
X_elo_test_t = torch.tensor(X_elo_test, dtype=torch.float32)
X_xgb_test_t = torch.tensor(X_xgb_test_pred, dtype=torch.float32)  
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Create datasets and data loaders
train_dataset = TensorDataset(X_home_train_t, X_away_train_t, X_elo_train_t, X_xgb_train_t, y_train_t)
valid_dataset = TensorDataset(X_home_valid_t, X_away_valid_t, X_elo_valid_t, X_xgb_valid_t, y_valid_t)
test_dataset = TensorDataset(X_home_test_t, X_away_test_t, X_elo_test_t, X_xgb_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#Define LSTM and ELO model with parameters
model = DualLSTMWithEloXGBoost(
        lstm_input_size=len(lstm_features),
        lstm_hidden_size=64,
        lstm_layers=5,
        mlp_input_size=len(elo_features),
        mlp_hidden_size=16,
        xgb_input_size=X_xgb_train_pred.shape[1],  
        num_classes=3,
        dropout=0.2
    ).to(device)

#Class weights for imbalanced dataset
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

num_epochs = 20
#Loss function and optimiser
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.7, weight_decay=0.01)

best_val_acc = 0
best_model_path = "best_model.pth"

for epoch in range(num_epochs):
    #Training the model
    model.train()
    running_loss = 0.0

    for X_home_b, X_away_b, X_elo_b, X_xgb_b, y_b in train_loader:
        X_home_b, X_away_b, X_elo_b, X_xgb_b, y_b = (
            X_home_b.to(device), X_away_b.to(device), X_elo_b.to(device), X_xgb_b.to(device), y_b.to(device)
        )
        optimizer.zero_grad()
        outputs = model(X_home_b, X_away_b, X_elo_b, X_xgb_b)
        loss = criterion(outputs, y_b)
        loss.backward()

        optimizer.step()
        #Calculate training loss
        running_loss += loss.item()

    # Compute Validation Accuracy
    avg_loss = running_loss / len(train_loader)
    #Validation stage
    model.eval()
    correct, total = 0, 0
    val_running_loss = 0.0
    with torch.no_grad():
        for X_home_b, X_away_b, X_elo_b, X_xgb_b, y_b in valid_loader:
            X_home_b, X_away_b, X_elo_b, X_xgb_b, y_b = (
                X_home_b.to(device), X_away_b.to(device), X_elo_b.to(device), X_xgb_b.to(device), y_b.to(device)
            )
    
            val_outputs = model(X_home_b, X_away_b, X_elo_b, X_xgb_b)  
            val_loss = criterion(val_outputs, y_b)  
            val_running_loss += val_loss.item()     
            
            _, val_preds = torch.max(val_outputs, 1)
            correct += (val_preds == y_b).sum().item()
            total += y_b.size(0)
    
    val_acc = correct / total
    avg_val_loss = val_running_loss / len(valid_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Save Best Model
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

print(f"Training complete! Best Validation Accuracy: {best_val_acc:.4f}")

# Load Best Model and Evaluate on Test Set
model.load_state_dict(torch.load(best_model_path))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_home_b, X_away_b, X_elo_b, X_xgb_b, y_b in test_loader:
        X_home_b, X_away_b, X_elo_b, X_xgb_b, y_b = (
            X_home_b.to(device), X_away_b.to(device), X_elo_b.to(device), X_xgb_b.to(device), y_b.to(device))

        test_outputs = model(X_home_b, X_away_b, X_elo_b, X_xgb_b) 
        _, test_preds = torch.max(test_outputs, 1)

        correct += (test_preds == y_b).sum().item()
        total += y_b.size(0)

test_acc = correct / total
print(f"Final Test Accuracy: {test_acc:.4f}")


test_leagues = df_test["League"].reset_index(drop=True)

league_correct = {}
league_total = {}

with torch.no_grad():
    for i in range(len(test_dataset)):
        league = test_leagues.iloc[i]

        X_home_b = X_home_test_t[i].unsqueeze(0).to(device)
        X_away_b = X_away_test_t[i].unsqueeze(0).to(device)
        X_elo_b = X_elo_test_t[i].unsqueeze(0).to(device)
        X_xgb_b = X_xgb_test_t[i].unsqueeze(0).to(device)
        y_b = y_test_t[i].unsqueeze(0).to(device)

        outputs = model(X_home_b, X_away_b, X_elo_b, X_xgb_b)
        _, preds = torch.max(outputs, 1)

        correct = (preds == y_b).item()

        if league not in league_correct:
            league_correct[league] = 0
            league_total[league] = 0

        league_correct[league] += correct
        league_total[league] += 1

print("\n=== Per-League Accuracy ===")
for league in league_correct:
    acc = league_correct[league] / league_total[league]
    print(f"{league}: {acc:.4f}")



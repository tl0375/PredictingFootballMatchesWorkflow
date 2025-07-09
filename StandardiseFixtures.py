import pandas as pd
from fuzzywuzzy import process
import re

# Load dataset and fixtures
dataset = pd.read_csv("dataset.csv")
fixtures = pd.read_csv("fixtures.csv")

# Extract unique team names from dataset (standardised names)
standard_names = set(dataset["Home_Team"].unique()).union(set(dataset["Away_Team"].unique()))

# Manual overrides
manual_team_map = {
    "1. FC Köln" : "FC Koln",
    "Borussia Mönchengladbach" : "M'gladbach",
    "Club Atlético de Madrid" : "Ath Madrid",
    "Deportivo Alavés" : "Alaves",
    "Eintracht Frankfurt" : "Ein Frankfurt",
    "FC Bayern München" : "Bayern Munich",
    "FC St. Pauli 1910" : "St Pauli",
    "Nottingham Forest FC" : "Nott'm Forest",
    "Paris Saint-Germain FC" : "Paris SG",
    "Stade Rennais FC 1901" : "Rennes",
    "Wolverhampton Wanderers FC" : "Wolves",
    "AC Pisa 1909" : "Pisa",
    "RCD Espanyol de Barcelona" : "Espanol"
}

# Threshold for fuzzy match acceptance
SIMILARITY_THRESHOLD = 90

# Clean function: removes trailing 'FC' with or without a space
def clean_team_name(name):
    return re.sub(r'\s*FC$', '', name, flags=re.IGNORECASE).strip()

# Build mapping from fixtures names to standard names
team_name_map = {}
fixtures_teams = set(fixtures["Home Team"].unique()).union(set(fixtures["Away Team"].unique()))

print("\nBuilding team name mapping (manual + fuzzy + Premier League FC cleanup)...")

# For each unique team name in fixtures
for team in fixtures_teams:
    # Find all rows where the team appears (as Home or Away)
    appears_in_pl = (
        ((fixtures["Home Team"] == team) | (fixtures["Away Team"] == team))
        & (fixtures["Competition"] == "Premier League")
    ).any()

    # Only remove "FC" if the team appears in a Premier League fixture - particular problem with this dataset solved
    team_for_matching = clean_team_name(team) if appears_in_pl else team

    if team in manual_team_map:
        team_name_map[team] = manual_team_map[team]
        print(f"✅ [Manual] {team} → {manual_team_map[team]}")
    else:
        match, score = process.extractOne(team_for_matching, list(standard_names))
        if score >= SIMILARITY_THRESHOLD:
            team_name_map[team] = match
            #print(f"✓ [Fuzzy Match] {team} → {match} (score: {score})")
        else:
            team_name_map[team] = team
            #print(f"[LOW CONFIDENCE] {team} → {match} (score: {score}) — no change made")

# Apply mapping to fixtures
fixtures["Home Team"] = fixtures["Home Team"].map(team_name_map)
fixtures["Away Team"] = fixtures["Away Team"].map(team_name_map)

# Final validation
mapped_fixtures_teams = set(fixtures["Home Team"].unique()).union(set(fixtures["Away Team"].unique()))
unmatched = mapped_fixtures_teams - standard_names

if unmatched:
    print("\nUnmatched team names in fixtures after standardisation:")
    for team in sorted(unmatched):
        print(" -", team)
else:
    print("\nAll team names in fixtures now match those in dataset.")


# Parse the datetime string in the 'Date' column
fixtures["Date"] = pd.to_datetime(fixtures["Date"], utc=True, errors='coerce')

# Create new 'Date' and 'Time' columns
fixtures["Time"] = fixtures["Date"].dt.strftime("%H:%M")
fixtures["Date"] = fixtures["Date"].dt.strftime("%Y-%m-%d")

# Move 'Date' and 'Time' to the front
cols = ["Date", "Time"] + [col for col in fixtures.columns if col not in ["Date", "Time"]]
fixtures = fixtures[cols]

fixtures.rename(columns={
    "Home Team": "Home_Team",
    "Away Team": "Away_Team"
}, inplace=True)

competition_map = {
    "Premier League": "E0",
    "Serie A"       : "I1",
    "Ligue 1"       : "F1",
    "La Liga"       : "SP1",
    "Bundesliga"    : "D1"
}

# Save the standardised fixtures
fixtures.to_csv("fixtures_clean.csv", index=False)
print("\nStandardised fixtures saved to 'fixtures_clean.csv'.")


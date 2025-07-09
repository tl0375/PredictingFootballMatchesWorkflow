import requests
import pandas as pd

API_KEY = "4231776c5a7f4ac4a0c749bc717bea24"  #API key

# Competition codes for top 5 leagues on football-data.org
COMPETITIONS = {
    "PL": "Premier League",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "PD": "La Liga",
    "BL1": "Bundesliga"
}

def get_upcoming_fixtures(competition_code):
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches?status=SCHEDULED"
    headers = {
        "X-Auth-Token": API_KEY
    }
    response = requests.get(url, headers=headers)
    data = response.json()

    fixtures = []
    for match in data.get("matches", []):
        fixture = {
            "Date": match["utcDate"],
            "Home Team": match["homeTeam"]["name"],
            "Away Team": match["awayTeam"]["name"],
            "Competition": COMPETITIONS.get(competition_code, competition_code),
            "Matchweek": match.get("matchday")
        }
        fixtures.append(fixture)

    return fixtures


if __name__ == "__main__":
    all_fixtures = []
    for code in COMPETITIONS.keys():
        print(f"Fetching fixtures for {COMPETITIONS[code]}...")
        league_fixtures = get_upcoming_fixtures(code)
        all_fixtures.extend(league_fixtures)

    if all_fixtures:
        df = pd.DataFrame(all_fixtures)
        df.to_csv("fixtures.csv", index=False)
        print(f"Saved {len(all_fixtures)} fixtures to fixtures.csv")
    else:
        print("No fixtures found.")


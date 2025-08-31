const maxMatchweeks = {
  premierleague: 38,
  laliga: 38,
  seriea: 38,
  bundesliga: 34,
  ligue1: 34
};

function getLeagueKey(leagueName) {
  // Converts "Premier League" -> "premierleague", etc.
  return leagueName.toLowerCase().replace(/ /g, '');
}

let currentMatchweek = {}; // will be populated from CSV
let allFixturesByLeague = {}; // already used in your code

async function initializeMatchweeksFromCSV() {
  const response = await fetch("Results.csv");
  const text = await response.text();

  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");

  const fixtures = lines.slice(1).map(line => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((h, i) => [h.trim(), values[i]?.trim()]));
  });

  // Get minimum matchweek per league
  const matchweeksByLeague = {};

  fixtures.forEach(fixture => {
    const league = fixture["League"];
    const week = parseInt(fixture["Matchweek"]);

    if (!matchweeksByLeague[league] || week < matchweeksByLeague[league]) {
      matchweeksByLeague[league] = week;
    }
  });

  currentMatchweek = matchweeksByLeague;

  // Store fixtures in cache (so loadFixtures doesn't need to refetch)
  for (const league of Object.keys(matchweeksByLeague)) {
    allFixturesByLeague[league] = fixtures.filter(f => f["League"] === league);
  }
}

async function loadFixtures(leagueName) {
  // Only fetch CSV once per league
  if (!allFixturesByLeague[leagueName]) {
    const response = await fetch("Results.csv");
    const text = await response.text();

    const lines = text.trim().split("\n");
    const headers = lines[0].split(",");

    const fixtures = lines.slice(1).map(line => {
      const values = line.split(",");
      return Object.fromEntries(headers.map((h, i) => [h.trim(), values[i]?.trim()]));
    });

    allFixturesByLeague[leagueName] = fixtures.filter(f => f["League"] === leagueName);
  }

  const leagueFixtures = allFixturesByLeague[leagueName];
  const containerId = 'fixtures-' + leagueName.toLowerCase().replace(/ /g, '');
  const container = document.getElementById(containerId);

  if (!container) {
    console.warn(`Fixtures container for league '${leagueName}' not found.`);
    return;
  }

  container.innerHTML = "";

  // Filter fixtures by current matchweek for this league
  // Make sure currentMatchweek is integer for filtering
  const cw = Math.floor(currentMatchweek[leagueName]);
  const matchweekFixtures = leagueFixtures.filter(f => parseInt(f["Matchweek"]) === cw);

  // If no fixtures for current matchweek, show all fixtures for league
  const fixturesToShow = matchweekFixtures.length > 0 ? matchweekFixtures : leagueFixtures;

  fixturesToShow.forEach(fixture => {
    const homeTeam = fixture["Home_Team"];
    const awayTeam = fixture["Away_Team"];
    const date = fixture["Date"];

    const probs = [
      { key: "home", val: parseFloat(fixture["Prob_Home_Win"]) || 0 },
      { key: "draw", val: parseFloat(fixture["Prob_Draw"]) || 0 },
      { key: "away", val: parseFloat(fixture["Prob_Away_Win"]) || 0 }
    ];

    // Convert to percentages (not rounded yet)
    probs.forEach(p => p.val *= 100);

    // Floor values and track remainders
    let total = 0;
    probs.forEach(p => {
      p.floorVal = Math.floor(p.val);
      p.remainder = p.val - p.floorVal;
      total += p.floorVal;
    });

    // Distribute leftover points
    let leftover = 100 - total;
    probs.sort((a, b) => b.remainder - a.remainder);
    for (let i = 0; i < leftover; i++) {
      probs[i % probs.length].floorVal += 1;
    }

    // Assign back to original variable names
    const homeProb = (probs.find(p => p.key === "home").floorVal) / 100;
    const drawProb = (probs.find(p => p.key === "draw").floorVal) / 100;
    const awayProb = (probs.find(p => p.key === "away").floorVal) / 100;

    const homeLogo = `logos/teams/${leagueName}/${homeTeam}.png`;
    const awayLogo = `logos/teams/${leagueName}/${awayTeam}.png`;

    const homeColor = getTeamColor(homeTeam);
    const awayColor = getTeamColor(awayTeam);

    const fixtureDiv = document.createElement("div");
    fixtureDiv.className = "fixture";

    fixtureDiv.innerHTML = `
      <div class="team-container">
        <div class="team">
          <img class="logo" src="${homeLogo}" alt="${homeTeam}" />
          <span class="team-name">${homeTeam}</span>
        </div>
        <div class="vs">vs</div>
        <div class="team" style="justify-content: flex-end;">
          <span class="team-name" style="margin-right:10px;">${awayTeam}</span>
          <img class="logo" src="${awayLogo}" alt="${awayTeam}" />
        </div>
      </div>
      <div class="date">${date}</div>
      <div class="bar">
        <div class="segment first-segment" style="width:${100 * homeProb}%; background:${homeColor.background}; color:${homeColor.text};" data-tooltip="${homeTeam} win: ${Math.round(100 * homeProb)}%">${Math.round(100 * homeProb)}%</div>
        <div class="segment middle-segment" style="width:${100 * drawProb}%; background:#ccc; color:white;">${Math.round(100 * drawProb)}%</div>
        <div class="segment last-segment" style="width:${100 * awayProb}%; background:${awayColor.background}; color:${awayColor.text};">${Math.round(100 * awayProb)}%</div>
      </div>
    `;

    container.appendChild(fixtureDiv);
  });

  // Update matchweek label inside the page for this league
  const label = document.getElementById(`currentMatchweek-${leagueName.toLowerCase().replace(/ /g, '')}`);
  if (label) {
    label.textContent = currentMatchweek[leagueName];
  }
}

function getTeamColor(team) {
  const colors = {
  "Arsenal": { background: "#EF0107", text: "white" },
  "Aston Villa": { background: "#670E36", text: "white" },
  "Bournemouth": {
    background: "repeating-linear-gradient(90deg, #DA291C, #DA291C 60px, black 60px, black 120px)",
    text: "white"
  },
  "Brentford": {
    background: "repeating-linear-gradient(90deg, #DA291C, #DA291C 60px, white 60px, white 120px)",
    text: "white"
  },
  "Brighton": {
    background: "repeating-linear-gradient(90deg, #0057B8, #0057B8 60px, white 60px, white 120px)",
    text: "white"
  },
  "Burnley": { background: "#6C1D45", text: "white" },
  "Chelsea": { background: "#034694", text: "white" },
  "Crystal Palace": {
    background: "repeating-linear-gradient(90deg, #1B458F, #1B458F 60px, #DA291C 60px, #DA291C 120px)",
    text: "white"
  },
  "Everton": { background: "#003399", text: "white" },
  "Fulham": { background: "white", text: "white" },
  "Leeds": { background: "#FFCD00", text: "white" },
  "Liverpool": { background: "#DA291C", text: "white" },
  "Manchester City": { background: "#6CABDD", text: "white" },
  "Manchester Utd": { background: "#DA291C", text: "white" },
  "Newcastle": {
    background: "repeating-linear-gradient(90deg, black, black 60px, white 60px, white 120px)",
    text: "white"
  },
  "Nott'm Forest": { background: "#DD0000", text: "white" },
  "Sunderland": {
    background: "repeating-linear-gradient(90deg, #FF0000, #FF0000 60px, white 60px, white 120px)",
    text: "white"
  },
  "Tottenham": { background: "white", text: "white" },
  "West Ham": { background: "#7A263A", text: "white" },
  "Wolves": { background: "#FDB913", text: "white" },


  "Alaves":{
    background: "repeating-linear-gradient(90deg, #005BAB, #005BAB 60px, white 60px, white 120px)",
    text: "white"
  },
  "Athletic Club": {
    background: "repeating-linear-gradient(90deg, #D91023, #D91023 60px, white 60px, white 120px)",
    text: "white"
  },
  "Ath Madrid": {
    background: "repeating-linear-gradient(90deg, #D62027, #D62027 60px, white 60px, white 120px)",
    text: "white"
  },
  "Barcelona": {
    background: "repeating-linear-gradient(90deg, #A50044, #A50044 40px, #004D98 40px, #004D98 80px)",
    text: "white"
  },
  "Celta": { background: "#A4C9E1", text: "white" },
  "Elche": { background: "#007A33", text: "white" },
  "Espanol": {
    background: "repeating-linear-gradient(90deg, #005BAB, #005BAB 60px, white 60px, white 120px)",
    text: "white"
  },
  "Getafe": { background: "#0052A5", text: "white" },
  "Girona": {
    background: "repeating-linear-gradient(90deg, #D62027, #D62027 60px, white 60px, white 120px)",
    text: "white"
  },
  "Levante": {
    background: "repeating-linear-gradient(90deg, #3A2483, #3A2483 60px, #E52935 60px, #E52935 120px)",
    text: "white"
  },
  "Mallorca": { background: "#D7000F", text: "white" },
  "Osasuna": { background: "#CE1126", text: "white" },
  "Vallecano": {
    background: "white",
    text: "white",
    // Their kit is white with a red diagonal stripe, hard to do in CSS background, simplified here.
  },
  "Betis": {
    background: "repeating-linear-gradient(90deg, #13754E, #13754E 60px, white 60px, white 120px)",
    text: "white"
  },
  "Real Madrid": { background: "white", text: "white" },
  "Oviedo": { background: "#1E40A4", text: "white" },
  "Sociedad": {
    background: "repeating-linear-gradient(90deg, #005BAB, #005BAB 60px, white 60px, white 120px)",
    text: "white"
  },
  "Sevilla": { background: "#white", text: "white" },
  "Valencia": { background: "white", text: "white" },
  "Villarreal": { background: "#FFDD00", text: "white" },

  "Atalanta": {
    background: "repeating-linear-gradient(90deg, #005CAF, #005CAF 60px, #000000 60px, #000000 120px)",
    text: "white"
  },
  "Bologna": {
    background: "repeating-linear-gradient(90deg, #C2002A, #C2002A 60px, #002147 60px, #002147 120px)",
    text: "white"
  },
  "Cagliari": {
    background: "repeating-linear-gradient(90deg, #8F143A, #8F143A 60px, #002147 60px, #002147 120px)",
    text: "white"
  },
  "Como": { background: "#0060A9", text: "white" },
  "Cremonese": { background: "#8C1919", text: "white" },
  "Fiorentina": { background: "#4B2E83", text: "white" },
  "Genoa": {
    background: "repeating-linear-gradient(90deg, #AE0000, #AE0000 60px, #003F87 60px, #003F87 120px)",
    text: "white"
  },
  "Hellas Verona": { background: "#034EA2", text: "white" },
  "Inter": {
    background: "repeating-linear-gradient(90deg, #004C9A, #004C9A 60px, #000000 60px, #000000 120px)",
    text: "white"
  },
  "Juventus": {
    background: "repeating-linear-gradient(90deg, white, white 60px, black 60px, black 120px)",
    text: "white"
  },
  "Lazio": { background: "#77C9F1", text: "white" },
  "Lecce": {
    background: "repeating-linear-gradient(90deg, #E31A24, #E31A24 60px, yellow 60px, yellow 120px)",
    text: "white"
  },
  "Milan": {
    background: "repeating-linear-gradient(90deg, #EA1C2D, #EA1C2D 60px, #000000 60px, #000000 120px)",
    text: "white"
  },
  "Napoli": { background: "#0092D2", text: "white" },
  "Parma": {
    background: "repeating-linear-gradient(90deg, #2C599D, #2C599D 60px, yellow 60px, yellow 120px)",
    text: "white"
  },
  "Pisa": {
    background: "repeating-linear-gradient(90deg, #000000, #000000 60px, #002147 60px, #002147 120px)",
    text: "white"
  },
  "Roma": { background: "#A1282A", text: "white" },
  "Sassuolo": {
    background: "repeating-linear-gradient(90deg, #0C6B35, #0C6B35 60px, #000000 60px, #000000 120px)",
    text: "white"
  },
  "Torino": { background: "#800000", text: "white" },
  "Udinese": {
    background: "repeating-linear-gradient(90deg, black, black 60px, white 60px, white 120px)",
    text: "white"
  },
  "Verona": {
    background: "repeating-linear-gradient(90deg, #003F87, #003F87 60px, yellow 60px, yellow 120px)",
    text: "white"
  },

  "Augsburg": { background: "white", text: "white" },
  "Heidenheim": {
    background: "repeating-linear-gradient(90deg, #DA291C, #DA291C 60px, #1B458F 60px, #1B458F 120px)",
    text: "white"
  },
  "FC Koln": { background: "white", text: "white" },
  "Union Berlin": { background: "#E2001A", text: "white" },
  "Leverkusen": { background: "#E10600", text: "white" },
  "Bayern Munich": { background: "#DC052D", text: "white" },
  "Dortmund": { background: "#FDE100", text: "white" },
  "M'gladbach": {
    background: "repeating-linear-gradient(90deg, #0C6B35, #0C6B35 60px, white 60px, white 120px)",
    text: "white"
  },
  "Mainz": { background: "#E2001A", text: "white" },
  "Ein Frankfurt": { background: "#000000", text: "white" },
  "St Pauli": { background: "#632A2A", text: "white" },
  "Hamburg": { background: "#0059B2", text: "white" },
  "RB Leipzig": { background: "white", text: "white" },
  "Freiburg": { background: "#DD1C26", text: "white" },
  "Hoffenheim": { background: "#005BAB", text: "white" },
  "Stuttgart": { background: "white", text: "white" },
  "Wolfsburg": { background: "#00A651", text: "white" },
  "Werder Bremen": { background: "#00602E", text: "white" },


  "Angers": {
    background: "repeating-linear-gradient(90deg, white, white 60px, black 60px, black 120px)",
    text: "white"
  },
  "Auxerre": { background: "#0033A0", text: "white" },
  "Brest": { background: "#E10600", text: "white" },
  "Le Havre": { background: "#0055A4", text: "white" },
  "Lens": {
    background: "repeating-linear-gradient(90deg, #E10600, #E10600 60px, #FFD700 60px, #FFD700 120px)",
    text: "white"
  },
  "Lille": { background: "#DA291C", text: "white" },
  "Lorient": { background: "#FF4200", text: "white" },
  "Lyon": { background: "white", text: "white" },
  "Marseille": { background: "#00BFFF", text: "white" },
  "Metz": { background: "#B71C1C", text: "white" },
  "Monaco": { background: "#E31E26", text: "white" },
  "Nantes": { background: "#FFD700", text: "white" },
  "Nice": {
    background: "repeating-linear-gradient(90deg, #DA291C, #DA291C 60px, black 60px, black 120px)",
    text: "white"
  },
  "Paris FC": { background: "#003366", text: "white" },
  "Paris SG": { background: "#004170", text: "white" },
  "Rennes": { background: "#C8102E", text: "white" },
  "Strasbourg": { background: "#005BAB", text: "white" },
  "Toulouse": { background: "#660066", text: "white" }
};

  return colors[team] || { background: "#999", text: "white" };
}

document.addEventListener("DOMContentLoaded", () => {
  const dropdown = document.querySelector(".dropdown");
  const toggleButton = document.getElementById("dropdownToggle");

  toggleButton.addEventListener("click", (e) => {
    e.stopPropagation(); // prevent bubbling
    dropdown.classList.toggle("open");
  });

  // Close dropdown if clicked outside
  document.addEventListener("click", () => {
    dropdown.classList.remove("open");
  });
});

async function showPage(id) {
  document.querySelectorAll('.page').forEach(page => page.style.display = 'none');

  const targetPage = document.getElementById(id);
  if (targetPage) targetPage.style.display = 'block';

  const titleTextEl = document.getElementById('navTitleText');
  const iconLeft = document.getElementById('navIconLeft');
  const iconRight = document.getElementById('navIconRight');
  const matchweekBar = document.getElementById('matchweek-bar');

  if (!titleTextEl || !iconLeft || !iconRight || !matchweekBar) {
    console.error("Nav elements not found");
    return;
  }

  const leagues = {
    premierleague: "Premier League",
    laliga: "La Liga",
    seriea: "Serie A",
    bundesliga: "Bundesliga",
    ligue1: "Ligue 1"
  };

  if (leagues[id]) {
    const leagueName = leagues[id];
    currentLeague = leagueName;

    titleTextEl.textContent = leagueName;
    iconLeft.src = `logos/countries/${getCountryCode(leagueName)}.png`;
    iconRight.src = `logos/leagues/${leagueName}.png`;
    iconLeft.style.display = "inline-block";
    iconRight.style.display = "inline-block";

    // Show sticky matchweek bar
    matchweekBar.classList.remove("hidden");

    await loadFixtures(leagueName);
    updateMatchweekLabel(leagueName);
    setupGlobalMatchweekNav(leagueName);

    // Highlight dropdown
    document.querySelectorAll('.dropdown-menu li').forEach(li => {
      li.classList.toggle('active', li.textContent.trim().includes(leagueName));
    });

  } else {
    // Not a league page (e.g. "about")
    currentLeague = null;

    titleTextEl.textContent = id.charAt(0).toUpperCase() + id.slice(1);
    iconLeft.style.display = "none";
    iconRight.style.display = "none";

    // Hide sticky matchweek bar
    matchweekBar.classList.add("hidden");

    // Remove highlight
    document.querySelectorAll('.dropdown-menu li').forEach(li => li.classList.remove('active'));
  }
}

function updateMatchweekLabel(leagueName) {
  const containerId = 'fixtures-' + leagueName.toLowerCase().replace(/ /g, '');
  const label = document.getElementById(`currentMatchweek-${leagueName.toLowerCase().replace(/ /g, '')}`);
  if (label) {
    label.textContent = currentMatchweek[leagueName];
  }
}

function setupGlobalMatchweekNav(leagueName) {
  const bar = document.getElementById('matchweek-bar');
  const label = document.getElementById('matchweek-label');
  const prevBtn = document.getElementById('prevWeekBtn');
  const nextBtn = document.getElementById('nextWeekBtn');

  const leagueKey = getLeagueKey(leagueName);
  const maxWeek = maxMatchweeks[leagueKey] || 38;

  bar.classList.remove("hidden"); // Show the nav

  function update() {
    label.textContent = `Matchweek ${currentMatchweek[leagueName]}`;
    prevBtn.disabled = currentMatchweek[leagueName] <= 1;
    nextBtn.disabled = currentMatchweek[leagueName] >= maxWeek;
  }

  prevBtn.onclick = async () => {
    if (currentMatchweek[leagueName] > 1) {
      currentMatchweek[leagueName]--;
      await loadFixtures(leagueName);
      updateMatchweekLabel(leagueName);
      update();
    }
  };

  nextBtn.onclick = async () => {
    if (currentMatchweek[leagueName] < maxWeek) {
      currentMatchweek[leagueName]++;
      await loadFixtures(leagueName);
      updateMatchweekLabel(leagueName);
      update();
    }
  };

  update();
}


function getCountryCode(leagueName) {
  switch (leagueName) {
    case "Premier League": return "eng";
    case "La Liga": return "esp";
    case "Serie A": return "ita";
    case "Bundesliga": return "ger";
    case "Ligue 1": return "fra";
    default: return "eng";
  }
}

fetch("last_updated.txt")
    .then(response => response.text())
    .then(text => {
      document.getElementById("last-updated").textContent = text;
    })
    .catch(() => {
      document.getElementById("last-updated").textContent = "Not available";
    });

window.onload = async () => {
  await initializeMatchweeksFromCSV();
  showPage('about');
};

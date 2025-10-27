# 🏀 HoopSim: Predicting NBA Champions with Data and AI

**Duration:** 4 weekends
**Goal:** Build a full NBA season simulator that predicts which team is most likely to win the championship using real data and probability models.

---

## 🎯 Overview

HoopSim is a data-driven basketball analytics project.  
You’ll use **real NBA data** to simulate an entire basketball season and estimate which team is most likely to win the championship.

By the end of this project, you’ll have:

- 🧮 A working simulator that plays every game in a season thousands of times  
- 📊 Visualizations showing each team’s championship odds  
- 🗣️ A short presentation explaining your findings and approach  

---

## 🧰 Tech Setup

**Tools:**
- Python 3 (Anaconda or Google Colab)
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tqdm`, `basketball-reference-scraper`
- Optional: `nba_api`, `streamlit`

**Install:**

```bash
pip install pandas numpy matplotlib scikit-learn tqdm basketball-reference-scraper
# optional
pip install nba_api streamlit
```

---

# 🗓️ Week 1 — Explore Real NBA Data & Understand Elo Ratings

### 🎯 Learning Goal
Understand how sports data is structured and how **Elo ratings** measure team strength.

### Step 1: Download Historical Game Data
Use [FiveThirtyEight’s NBA Elo dataset](https://github.com/fivethirtyeight/data/tree/master/nba-elo).

```python
import pandas as pd
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
nba = pd.read_csv(url)
nba.head()
```

### Step 2: Clean and Filter One Season

```python
season = 2023  # choose a past season
season_data = nba[nba['year_id'] == season].copy()
season_data[['team_id','opp_id','pts','opp_pts','elo_i','elo_n']].head()
```

### Step 3: Calculate End-of-Season Elo per Team

```python
last_games = season_data.sort_values(['team_id','date_game']).groupby('team_id').tail(1)
last_elo = last_games[['team_id','elo_n']].rename(columns={'team_id':'team','elo_n':'last_elo'})
```

### Step 4: Visualize

```python
import matplotlib.pyplot as plt
top = last_elo.sort_values('last_elo', ascending=False).head(10)
plt.barh(top['team'], top['last_elo'])
plt.gca().invert_yaxis()
plt.xlabel("End of Season Elo")
plt.title(f"Top Teams — {season}")
plt.show()
```

#### 💭 Reflection
- What does Elo measure?
- Which teams surprised you in this season?
- What might Elo miss (injuries, trades, luck)?

---

# 🗓️ Week 2 — Estimate Team Strength for the New Season (Hybrid Model)

### 🎯 Learning Goal
Build this season’s starting ratings using both **last year’s performance** and **this year’s roster**.

### Step 1: Gather Player Data (Basketball-Reference Scraper)

```python
from basketball_reference_scraper.teams import get_roster_stats

teams_abr = ['ATL','BOS','BRK','CHI','CHO','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM',
             'MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','TOR','UTA','WAS']

season_end_year = 2024
rosters = []
for t in teams_abr:
    df = get_roster_stats(t, season_end_year=season_end_year)
    df['TEAM'] = t
    rosters.append(df)
roster = pd.concat(rosters, ignore_index=True)
```

### Step 2: Compute Each Team’s Average Player Rating (e.g., BPM)

```python
roster['MP'] = pd.to_numeric(roster.get('MP', 0), errors='coerce').fillna(0)
eligible = roster[roster['MP'] >= 200].copy()
eligible['BPM'] = pd.to_numeric(eligible.get('BPM', 0), errors='coerce').fillna(0)
player_strength = eligible.groupby('TEAM')['BPM'].mean().reset_index()
player_strength.columns = ['team','player_rating']
```

### Step 3: Regress Last Season’s Elo Toward League Average & Blend

```python
LEAGUE_AVG = 1500
W_DECAY = 0.35  # regression factor

last_elo['regressed_elo'] = (1 - W_DECAY) * last_elo['last_elo'] + W_DECAY * LEAGUE_AVG

hybrid = pd.merge(last_elo, player_strength, on='team', how='inner')
hybrid['start_rating'] = 0.5 * hybrid['regressed_elo'] + 0.5 * hybrid['player_rating']
hybrid['start_rating'] = 1500 + (hybrid['start_rating'] - hybrid['start_rating'].mean())
```

### Step 4: Save & Plot

```python
hybrid.sort_values('start_rating', ascending=False).head(10)
hybrid.to_csv('data/team_start_ratings.csv', index=False)

import matplotlib.pyplot as plt
top = hybrid.sort_values('start_rating', ascending=False).head(10)
plt.barh(top['team'], top['start_rating'])
plt.gca().invert_yaxis()
plt.xlabel("Hybrid Rating (Elo + Player)")
plt.title("Estimated Team Strengths")
plt.show()
```

#### 💭 Reflection
- Which teams improved or weakened the most?  
- How sensitive are results to the BPM minutes cutoff?  
- What happens if you weight players by minutes?  

---

# 🗓️ Week 3 — Simulate the Full NBA Season

### 🎯 Learning Goal
Use your ratings to simulate every game of the season and estimate each team’s chance to win the Finals.

### Step 1: Get the Season Schedule (NBA Public API)

```python
import requests, pandas as pd
url = "https://data.nba.net/prod/v1/2024/schedule.json"
schedule = requests.get(url).json()
games = pd.json_normalize(schedule['league']['standard'])
games = games[['startDateEastern','homeTeam.teamTricode','awayTeam.teamTricode']].rename(
    columns={'startDateEastern':'date','homeTeam.teamTricode':'home','awayTeam.teamTricode':'away'})
```

### Step 2: Win Probability Function

```python
import numpy as np, random

def win_probability(home_elo, away_elo, hca=65):
    diff = (home_elo + hca) - away_elo
    return 1.0 / (1.0 + 10 ** (-diff / 400))

def simulate_game(row, ratings, hca=65):
    home, away = row['home'], row['away']
    p_home = win_probability(ratings[home], ratings[away], hca=hca)
    return 1 if random.random() < p_home else 0  # 1 = home wins
```

### Step 3: Simulate Regular Season

```python
def simulate_regular_season(games, start_ratings, hca=65):
    ratings = start_ratings.copy()
    wins = {team: 0 for team in ratings.keys()}
    for _, g in games.iterrows():
        res = simulate_game(g, ratings, hca=hca)
        if res == 1:
            wins[g['home']] += 1
        else:
            wins[g['away']] += 1
    return wins
```

### Step 4: Playoff Simulation (Best-of-7)

```python
def simulate_series(teamA, teamB, ratings, hca_teamA=True):
    winsA = winsB = 0
    home_order = [teamA, teamA, teamB, teamB, teamA, teamB, teamA]
    for home in home_order:
        away = teamB if home == teamA else teamA
        p_home = win_probability(ratings[home], ratings[away], hca=65)
        if random.random() < p_home:
            if home == teamA: winsA += 1
            else: winsB += 1
        else:
            if home == teamA: winsB += 1
            else: winsA += 1
        if winsA == 4 or winsB == 4:
            break
    return teamA if winsA > winsB else teamB
```

### Step 5: Monte Carlo — Repeat Many Seasons

```python
from tqdm import trange

def simulate_full_season_once(games, ratings):
    wins = simulate_regular_season(games, ratings)
    teams_sorted = sorted(wins.keys(), key=lambda t: wins[t], reverse=True)
    bracket = teams_sorted[:8]
    r8 = [(bracket[0], bracket[7]), (bracket[3], bracket[4]), (bracket[1], bracket[6]), (bracket[2], bracket[5])]
    r4 = [simulate_series(a,b,ratings) for a,b in r8]
    r2 = [simulate_series(r4[0], r4[1], ratings), simulate_series(r4[2], r4[3], ratings)]
    champ = simulate_series(r2[0], r2[1], ratings)
    return champ

def monte_carlo_titles(games, start_ratings, n=10000):
    titles = {t:0 for t in start_ratings.keys()}
    for _ in trange(n):
        champ = simulate_full_season_once(games, start_ratings)
        titles[champ] += 1
    odds = pd.Series({t: titles[t]/n for t in titles}).sort_values(ascending=False)
    return odds
```

---

# 🗓️ Week 4 — Visualize, Test, and Present

### 🎯 Learning Goal
Present your simulation results like a professional data scientist.

### Step 1: Visualize Championship Odds

```python
top10 = odds.head(10)
top10.plot(kind='barh', color='orange')
import matplotlib.pyplot as plt
plt.xlabel("Chance to Win Championship")
plt.title("Simulated NBA Championship Odds")
plt.gca().invert_yaxis()
plt.show()
```

### Step 2: Backtest on Past Seasons
Try 2–3 previous seasons and check if your model’s **top 3 predictions** included the actual champion.

### Step 3: Sensitivity Tests

```python
for w_decay in [0.25, 0.35, 0.45]:
    # recompute hybrid start ratings and rerun Monte Carlo simulation
    pass
```

### Step 4: Final Slides / Report
**Recommended slide structure:**
1. Introduction — what question you asked  
2. Data sources (FiveThirtyEight, Basketball-Reference, NBA API)  
3. How the model works (Elo + player data + simulation)  
4. Results — top 10 championship odds  
5. Insights — surprises and key learnings  
6. Limitations — what your model didn’t include  
7. Next steps — future improvements  

---

## 📚 Data Sources

- [FiveThirtyEight NBA Elo dataset](https://github.com/fivethirtyeight/data/tree/master/nba-elo)  
- [Basketball Reference](https://www.basketball-reference.com/) — player stats and rosters  
- [NBA JSON API](https://data.nba.net/prod/v1/2024/schedule.json) — official schedules  
- [FiveThirtyEight RAPTOR ratings](https://github.com/fivethirtyeight/data/tree/master/nba-raptor) — player impact

---

## ✅ Final Checklist

- [x] Code runs end-to-end  
- [x] Uses real NBA data  
- [x] Produces clear charts and explanations  
- [x] Predicts a champion with reasoning  
- [x] Presentation summarizing approach and results  

---

## 🚀 Stretch Goals

- Build a Streamlit dashboard for live simulations  
- Add injury randomness  
- Simulate playoff brackets by conference  
- Compare Elo vs ML-based predictions  

---

## 🏆 Outcome

By the end of **HoopSim**, you will have:  
- A realistic NBA season simulator using real data and probability  
- A fun, impressive AI project to showcase on your resume or college portfolio  
- A clear understanding of how data science and sports analytics work together  

---

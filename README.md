# 🏀 HoopSim: Predicting NBA Champions with Data and AI

**Duration:** 4 weekends
**Goal:** Build a full NBA season simulator that predicts which team is most likely to win the championship using real data and probability models.

---

## 🚀 Quick Links

- **Requirements:** [requirements.txt](requirements.txt)  
- **Conda env:** [environment.yml](environment.yml)  
- **Quickstart Notebook:** [HoopSim_Quickstart.ipynb](HoopSim_Quickstart.ipynb)

---

## 🧰 Cross‑Platform Environment Setup

```bash
# 1) Create environment
conda env create -f environment.yml

# 2) Activate
conda activate hoopsim

# 3) Register Jupyter kernel
python -m ipykernel install --user --name hoopsim --display-name "Python (hoopsim)"
```

---

## 📓 Run the Quickstart Notebook

1) Open **HoopSim_Quickstart.ipynb** in Jupyter (VS Code, JupyterLab, or `jupyter notebook`).  
2) Make sure the kernel is **Python (hoopsim)**.  
3) Run cells top-to-bottom. The notebook will:
   - Verify your environment
   - Download the FiveThirtyEight Elo dataset
   - Compute end-of-season Elo for a chosen year
   - Build a **hybrid start rating** (regressed Elo + roster metric placeholder)
   - Save `data/team_start_ratings.csv` for simulation

---

## 🗂️ Project Structure (suggested)

```
HoopSim/
├─ data/
│  ├─ team_start_ratings.csv
│  └─ (raw/processed files you add)
├─ notebooks/
│  └─ (your analysis notebooks)
├─ src/
│  ├─ ratings.py
│  ├─ features.py
│  ├─ sim.py
│  └─ eval.py
├─ environment.yml
└─ README.md
```

---

## 🌐 Notes on Data Access

- The Quickstart pulls from **FiveThirtyEight’s NBA Elo** GitHub CSV. You need an internet connection.  
- For the roster-based hybrid rating, switch the placeholder to **Basketball-Reference Scraper** calls once you’re online.  
- To work offline, download CSVs ahead of time and update paths in the notebook.

---

## 🧪 Next Steps

- Implement season schedule download from `https://data.nba.net/prod/v1/<year>/schedule.json`  
- Write `simulate_regular_season` and `simulate_series` functions in `src/sim.py`  
- Run Monte Carlo (`10_000` runs) and plot championship odds  
- Backtest on 2–3 past seasons for validation

---

## ❓ Troubleshooting

- **Kernel not showing up in Jupyter:** Re-run `python -m ipykernel install --user --name hoopsim --display-name "Python (hoopsim)"` and restart Jupyter.  
- **Package conflicts in conda:** Try `mamba` or use **Option B (uv)**.  
- **SSL errors** when downloading data: try another network or download the CSVs manually and load from disk.

---

## 📜 License

Educational use. Cite data sources when publishing: FiveThirtyEight, Basketball-Reference, NBA API.

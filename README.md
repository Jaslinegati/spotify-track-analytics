# Spotify Track Analytics
### What makes a song go viral?

**Tools:** Python · pandas · scikit-learn · seaborn · K-Means  
**Dataset:** [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) — 89,741 tracks across 113 genres

---

## Business Problem

> *An A&R team at a record label wants to move beyond gut instinct when signing new artists. Can audio features alone predict whether a track will become a hit — and what should producers prioritise in the studio?*

---

## Key Findings

| Finding | Business Implication |
|---------|---------------------|
| Only **3.5%** of tracks reach popularity ≥70 | Signing decisions must be data-driven, not gut feel |
| **Genre** is the single strongest predictor of hit status | A&R teams should track which genres are trending each quarter |
| **Euphoric tracks** (high energy + high valence) outperform all other mood quadrants | Production briefs should target the top-right of the mood map |
| **Explicit tracks** score higher on average | Censoring for radio may actively hurt streaming numbers |
| **Acoustic/ambient** clusters at low popularity | These genres serve niche audiences — don't market them as mainstream |

---

## Model Performance

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.730 |
| **Random Forest** | **0.783** ✓ |

The Random Forest correctly separates hits from non-hits in ~78% of cases using audio features alone.

---

## Analysis Walkthrough

### Notebook 1 — Data Loading
Downloads 89k tracks from Hugging Face, inspects schema and data quality.

### Notebook 2 — Full Analysis

**Part 1 — Popularity Distribution**  
Classifies all tracks into 4 tiers: Undiscovered / Mid-tier / Popular / Hit. Compares explicit vs clean track performance.

**Part 2 — Audio Feature Correlations**  
Measures which of Spotify's 9 audio dimensions (danceability, energy, valence, tempo, etc.) correlate most strongly with popularity. Includes a full feature correlation heatmap.

**Part 3 — Genre Analysis**  
Ranks all 113 genres by average popularity and hit rate. Identifies which genres produce the most and fewest hits.

**Part 4 — The Mood Map**  
Plots 3,000 tracks on a Valence × Energy scatter coloured by popularity. Reveals four emotional quadrants and their average popularity scores.

![Mood Map](data/processed/mood_map.png)

**Part 5 — Hit Prediction Model**  
Trains Logistic Regression and Random Forest on 12 audio + metadata features. Compares AUC and extracts feature importance.

![Feature Importance](data/processed/hit_prediction_model.png)

**Part 6 — Song Clustering**  
K-Means clustering (k=5) groups songs by audio DNA without genre labels. Elbow method used to select k. Clusters labelled: Party Anthems, Acoustic/Chill, Instrumental/Ambient, Rap/Spoken Word, Mainstream Pop.

![Song Clusters](data/processed/song_clusters.png)

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/Jaslinegati/spotify-track-analytics

# 2. Create virtual environment
python3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# 3. Run notebooks in order
# Notebook 1 downloads the dataset automatically via Hugging Face
jupyter lab
```

---

*Prepared by Jasline Mwita — Data Analyst & Data Scientist*

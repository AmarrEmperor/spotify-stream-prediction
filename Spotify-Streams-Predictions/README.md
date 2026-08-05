<div align="center">

# Spotify Streams Prediction

### Cross-Platform Popularity Analysis and Machine Learning

A data-science project that explores what drives Spotify stream counts and evaluates whether cross-platform playlist and chart visibility can be used to predict a song's popularity.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Experimental-006600)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

</div>

---

## Project Overview

This project analyzes a dataset of **953 Spotify tracks and 24 variables** to understand the relationship between stream counts, platform exposure, release information, and audio characteristics.

The work follows an end-to-end machine-learning workflow:

1. Inspect and clean the raw dataset.
2. Convert inconsistent numeric values and handle missing data.
3. Explore feature correlations and cross-platform relationships.
4. Select meaningful feature vectors for prediction.
5. Train and evaluate linear regression models.
6. Prepare Random Forest and XGBoost experiments for nonlinear modeling.
7. Save trained models for later inference or application development.

> **Central finding:** playlist visibility is much more strongly associated with stream count than basic audio characteristics such as BPM, energy, valence, or acousticness.

---

## Research Questions

The analysis was designed around the following questions:

- Which available features have the strongest relationship with Spotify streams?
- Does popularity on Apple Music, Deezer, and Shazam help predict Spotify performance?
- Do playlist counts provide more predictive value than chart appearances?
- Can audio properties alone explain why a song becomes highly streamed?
- Does adding Spotify-specific platform indicators improve a cross-platform regression model?

---

## Dataset

The cleaned dataset contains **953 observations** and **24 columns**.

| Feature group | Examples |
|---|---|
| Track information | `track_name`, `artist(s)_name`, `artist_count` |
| Release information | `released_year`, `released_month`, `released_day` |
| Spotify exposure | `in_spotify_playlists`, `in_spotify_charts` |
| Cross-platform exposure | Apple, Deezer, and Shazam playlist/chart indicators |
| Audio characteristics | `bpm`, `danceability_%`, `energy_%`, `valence_%`, `acousticness_%` |
| Prediction target | `streams` |

The notebooks expect the raw data at:

```text
data/spotify-2023.csv
```

The cleaning workflow exports:

```text
data/spotify-2023-cleaned.csv
```

> The raw dataset is not included in this README package. Add it to the `data/` directory and follow the terms of its original source.

---

## Data Cleaning

The cleaning notebook performs the following operations:

- Loads the CSV using `latin-1` encoding.
- Inspects column names, data types, and missing values.
- Converts numeric-looking strings with `pandas.to_numeric(..., errors="coerce")`.
- Handles malformed or missing numeric values using mean imputation.
- Standardizes text fields and removes surrounding whitespace.
- Replaces missing categorical labels with `Unknown`.
- Confirms that the cleaned dataset contains no remaining missing values.
- Exports the final **953 × 24** dataset for analysis and modeling.

---

## Exploratory Data Analysis

### Correlation analysis

The strongest correlations with `streams` were found among platform-distribution variables:

| Feature | Correlation with streams |
|---|---:|
| `in_spotify_playlists` | **0.790** |
| `in_apple_playlists` | **0.772** |
| `in_deezer_playlists` | **0.516** |
| `in_apple_charts` | **0.320** |
| `in_spotify_charts` | **0.246** |
| `in_deezer_charts` | **0.229** |
| `in_shazam_charts` | **0.051** |

By comparison, the basic audio features showed little linear relationship with streams. For example, BPM was nearly uncorrelated with the target, while danceability, energy, valence, and acousticness all remained weak predictors.

<p align="center">
  <img src="../images/more_detailed_correlation_hm.png" alt="Detailed feature correlation heatmap" width="900">
</p>

### Interpretation

The heatmap suggests that **distribution and visibility matter more than a song's basic audio profile** in this dataset. Playlist placement on Spotify and Apple Music provides the clearest signal, while chart appearances show positive but weaker relationships.

This does **not** prove that playlists cause streams. Playlist counts and streams can influence each other, and both may be affected by factors that are not available in the dataset, including marketing, artist popularity, social-media trends, release timing, and editorial promotion.

---

## Cross-Platform Analysis

Chart presence shows a loose positive relationship with streams, but the scatter plots contain substantial variation and several outliers. This means that chart visibility alone is insufficient for accurate prediction.

| Spotify | Apple Music |
|---|---|
| ![Spotify charts versus streams](../images/charts%20vs%20stream%20graph/spotify_charts%20vs%20stream.png) | ![Apple charts versus streams](../images/charts%20vs%20stream%20graph/apple_charts%20vs%20stream.png) |

| Deezer | Shazam |
|---|---|
| ![Deezer charts versus streams](../images/charts%20vs%20stream%20graph/deezer_charts%20vs%20stream.png) | ![Shazam charts versus streams](../images/charts%20vs%20stream%20graph/shazam_charts%20vs%20stream.png) |

Key observations:

- Spotify and Apple chart indicators show modest upward trends.
- Deezer chart counts explain less of the variation in streams.
- Shazam chart presence is weakly related to total Spotify streams.
- Many tracks with low chart counts still have very high stream totals.
- Extreme hits create visible outliers and make the prediction problem difficult.

---

## Feature Engineering

Two main feature vectors were evaluated.

### 1. Cross-platform feature vector

This model intentionally excludes Spotify playlist and chart variables to test whether external platforms can predict Spotify streams.

```python
cross_platform_features = [
    "in_apple_playlists",
    "in_apple_charts",
    "in_deezer_charts",
    "in_deezer_playlists",
    "in_shazam_charts",
]
```

### 2. Strong platform feature vector

This model adds the most strongly correlated Spotify exposure variables.

```python
strong_features = [
    "in_spotify_playlists",
    "in_spotify_charts",
    "in_apple_playlists",
    "in_apple_charts",
    "in_deezer_charts",
    "in_deezer_playlists",
    "in_shazam_charts",
]
```

---

## Modeling

The completed benchmark uses `LinearRegression` from scikit-learn with:

- An **80/20 train-test split**
- `random_state=42`
- Root Mean Squared Error (**RMSE**)
- Coefficient of determination (**R²**)
- Serialized model output using `joblib`

### Confirmed model results

| Model | Feature set | RMSE | R² |
|---|---|---:|---:|
| Linear Regression | Cross-platform only | 325,390,533 | 0.609 |
| Linear Regression | Strong platform features | **285,656,334** | **0.699** |

Adding Spotify playlist and chart indicators improved the model from approximately **0.61 R² to 0.70 R²** and reduced prediction error.

<p align="center">
  <img src="../images/linear_regression_scatter.png" alt="Actual versus predicted Spotify streams" width="600">
</p>

The actual-versus-predicted plot shows that the model captures the broad popularity trend, but it still struggles with rare blockbuster tracks and other extreme values.

### Saved models

The completed notebook exports:

```text
models/spotify_cross_platform_lrmodel.pkl
models/spotify_strong_fvs_lrmodel.pkl
```

---

## Experimental Models

The project also includes an experimental notebook for:

- `RandomForestRegressor`
- `XGBRegressor`
- Nonlinear feature relationships
- Side-by-side RMSE and R² comparison

The model definitions and initial correlation workflow are present, but the final Random Forest/XGBoost benchmark has not yet been fully executed and validated in the current notebook version. Their metrics are therefore intentionally not reported here.

---

## 📁 Repository Structure

```text
spotify-stream-prediction/
│
├── data/
│   ├── spotify-2023.csv
│   └── spotify-2023-cleaned.csv
│
├── images/
│   ├── charts vs stream graph/
│   │   ├── apple_charts vs stream.png
│   │   ├── deezer_charts vs stream.png
│   │   ├── shazam_charts vs stream.png
│   │   └── spotify_charts vs stream.png
│   │
│   ├── correlation_heatmap.png
│   ├── linear_regression_scatter.png
│   └── more_detailed_correlation_hm.png
│
├── models/
│   ├── spotify_cross_platform_lrmodel.pkl
│   └── spotify_strong_fvs_lrmodel.pkl
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_correlation_analysis.ipynb
│   ├── 03_cross_platform_and_strong_fv_lrmodels.ipynb
│   ├── 04_randomforest_xgboost_models.ipynb
│   └── notebook.ipynb
│
└── README.md

```

`notebook.ipynb` contains the original exploratory work, while the numbered notebooks organize the project into a clearer reproducible pipeline.

---


## Notebook Guide

| Notebook | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Loads, inspects, cleans, validates, and exports the dataset |
| `02_feature_correlation_analysis.ipynb` | Examines correlations, heatmaps, scatter plots, and candidate feature vectors |
| `03_cross_platform_and_strong_fv_lrmodels.ipynb` | Trains, evaluates, and saves the two linear regression models |
| `04_randomforest_xgboost_models.ipynb` | Develops experimental tree-based regression models |
| `notebook.ipynb` | Preserves the initial exploratory analysis and early modeling experiments |

---

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd spotify-streams-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

### 4. Add the dataset

Place the raw CSV at:

```text
data/spotify-2023.csv
```

### 5. Run the notebooks in order

```bash
jupyter lab
```

Recommended execution order:

```text
01_data_cleaning.ipynb
02_feature_correlation_analysis.ipynb
03_cross_platform_and_strong_fv_lrmodels.ipynb
04_randomforest_xgboost_models.ipynb
```

---

## Key Skills Demonstrated

- Data inspection and preprocessing with pandas
- Missing-value handling and numeric type conversion
- Exploratory data analysis and visualization
- Correlation-based feature selection
- Regression model development with scikit-learn
- Model evaluation using RMSE and R²
- Cross-platform feature analysis
- Model serialization with joblib
- Initial Random Forest and XGBoost implementation
- Reproducible notebook organization

---

## Limitations

- Correlation does not establish causation.
- Playlist and chart exposure may be both a cause and a consequence of high streams.
- The dataset is relatively small for modeling global music popularity.
- Stream counts are highly skewed and contain extreme outliers.
- Artist reputation, marketing budget, geography, genre, social trends, and editorial support are not modeled.
- Mean imputation may reduce the natural variation of malformed or missing values.
- A single train-test split gives less reliable performance estimates than repeated cross-validation.
- The current regression models predict raw stream counts rather than a log-transformed target.

---

## Future Improvements

- Complete and validate the Random Forest and XGBoost benchmarks.
- Use cross-validation and hyperparameter tuning.
- Apply a logarithmic transformation to the stream target.
- Compare MAE, RMSE, R², and median absolute error.
- Add feature importance and SHAP-based interpretation.
- Investigate multicollinearity among playlist variables.
- Build artist-aware or release-year-aware validation splits.
- Add genre, social-media, regional, and temporal features.
- Package the best model behind a Streamlit or FastAPI application.
- Add automated tests and a reproducible `requirements.txt` file.

---

## Conclusion

The project demonstrates that cross-platform visibility provides meaningful predictive information about Spotify streams. The strongest linear model explains roughly **70% of the variation** in the held-out test set, with Spotify and Apple playlist counts acting as the most informative variables.

At the same time, the results show why music popularity is difficult to model: viral hits, artist-level effects, marketing, and cultural trends create patterns that cannot be fully captured by platform counts and basic audio features alone.

---

## Author

**Daniel HK**  
Computer Science Engineering Student | Data Science and Machine Learning

---

<div align="center">

If this project helped you, consider giving the repository a ⭐.

</div>

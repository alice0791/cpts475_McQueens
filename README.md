# cpts475_McQueens
CPTS 475 Data Science project analyzing and predicting U.S. traffic accident severity using weather, time, and road structure data.

## Traffic Accident Severity Prediction

This project analyzes U.S. traffic accident data from 2016–2023 to investigate how weather conditions, timeframe, and nearby road structure can be used to predict accident severity.

## Research Question

How can machine learning use weather conditions, timeframe, and nearby road structure to predict the severity of a traffic accident?

## Dataset

**US Accidents (2016–2023)**  
Source: Kaggle  
Approximately 7.7 million traffic accident records from 49 U.S. states.

## Planned Models

- Multinomial Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- Shallow Neural Network
- Wide Neural Network
- Deep Neural Network
- Regularized Deep Neural Network

## Team

- Alice Hui
- Emily
- Kenneth

## Course

CPTS 475 - Data Science

## Setup

```bash
conda create -n cpts475 python=3.11
conda activate cpts475
pip install -r requirements.txt
nbstripout --install
```

## Get the data

The CSV is about 3 GB, so it's not in the repo. Download it with the Kaggle CLI
(you need a token in `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download sobhanmoosavi/us-accidents -p data/raw --unzip
```

Or get the parquet files from the shared Google Drive and put them in `data/processed/`.

## Running it

```bash
pytest
python -m src.split
```

`src.split` cleans the data and saves these to `data/processed/`:

- `clean.parquet`: all cleaned rows
- `train.parquet`, `val.parquet`, `test.parquet`: 70/15/15 split, stratified on Severity
- `train_sample_1m.parquet`: 1M rows from train for slower models
- `raw.parquet`: copy of the raw CSV that loads faster

Example:

```python
import pandas as pd
from src import config

train = pd.read_parquet(config.TRAIN_PARQUET)
X = train[config.FEATURE_GROUPS["weather"] + config.FEATURE_GROUPS["time"]]
y = train[config.TARGET]
```

The splits are not scaled or imputed, so do that in a `Pipeline` fit on train only.
Use `src.evaluate.evaluate(model, X, y, name)` so everyone's results save the same way in `results/`.

## Data card

| | |
|---|---|
| Rows in raw CSV | TBD |
| Bad `Start_Time` | TBD |
| Missing `Sunrise_Sunset` | TBD |
| Exact duplicates | TBD |
| Same time and location | TBD |
| Rows after cleaning | TBD |
| Train / val / test | TBD |
| Severity 1 / 2 / 3 / 4 | TBD |

Dropped columns:
- Leakage: `End_Time`, `End_Lat`, `End_Lng`, `Distance(mi)`, `Description`
- IDs and constants: `ID`, `Country`, `Turning_Loop`
- Not used: `Street`, `City`, `County`, `Zipcode`, `Airport_Code`, `Weather_Timestamp`, `Timezone`, `Start_Lat`, `Start_Lng`
- Redundant: `Wind_Chill(F)`, the twilight columns

Missing values:
- `Precipitation(in)`: filled with 0, with a `precip_missing` flag
- Other weather numbers: left as NaN with a `*_missing` flag
- `Weather_Condition`, `Wind_Direction`: "Unknown"
- Rows missing `Start_Time` or `Sunrise_Sunset`: dropped

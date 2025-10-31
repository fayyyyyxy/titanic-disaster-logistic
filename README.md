# Titanic Disaster Prediction using Logistic Regression

*A reproducible class project for Northwestern University MLDS 400 (Data Engineering).*  
*Author: Xinyue (Fay) Yan*

This repo has two containers (Python and R) that:
1) train a simple logistic regression on `train.csv`,  
2) generate predictions for `test.csv`,  
3) write outputs to your local `src/data/` folder.

- Python app → `src/app/main.py` → **`src/data/pred_test_py.csv`**  
- R app → `src/rapp/main.R` → **`src/data/pred_test_r.csv`**  
Both apps mount `src/data/` at runtime (data is not baked into images and is git-ignored).

## 0) Prereqs
- **Docker Desktop** running
- **Git**
- **Kaggle** account to download data

## 1) Clone
```bash
git clone https://github.com/fayyyyyxy/titanic-disaster-logistic.git
cd titanic-disaster-logistic
```
### Repo layout

```text
.
├─ Dockerfile                 # Python container (root)
├─ requirements.txt           # Python deps
├─ src/
│  ├─ app/
│  │  └─ main.py              # Python script (prints steps + writes pred_test_py.csv)
│  ├─ data/                   # <put Kaggle CSVs here>  (git-ignored)
│  └─ rapp/
│     ├─ Dockerfile           # R container
│     ├─ install_packages.R   # R package install script
│     └─ main.R               # R script (prints steps + writes pred_test_r.csv)
└─ .dockerignore
```
## 2) Get the data (Kaggle)
1. Open https://www.kaggle.com/competitions/titanic/data 
2. Download **`train.csv`** and **`test.csv`**  
3. Place them here:

```bash
mkdir -p src/data
# move your downloaded files into:
# src/data/train.csv
# src/data/test.csv
```
## 3) Python container — build & run
### Build (from repo root)
```bash
docker build -t titanic-ml .
```
### Run
macOS / Linux
```bash
docker run --rm \
  -v "$(pwd)/src/data:/app/src/data" \
  titanic-ml
```
Windows (PowerShell)
```powershell
docker run --rm ^
  -v "${pwd}\src\data:/app/src/data" ^
  titanic-ml
```
**Expected output**

Console logs: load → preprocess → train accuracy → test prediction sample

File created: `src/data/pred_test_py.csv`

## 4) R container — build & run
### Build (Dockerfile lives in `src/rapp/`)
```bash
docker build -t titanic-r -f src/rapp/Dockerfile .
```
### Run
macOS / Linux
```bash
docker run --rm \
  -v "$(pwd)/src/data:/app/src/data" \
  titanic-r
```
Windows (PowerShell)
```powershell
docker run --rm ^
  -v "${pwd}\src\data:/app/src/data" ^
  titanic-r
```
**Expected output**

Console logs: load → preprocess → train accuracy → test prediction 

File created: `src/data/pred_test_r.csv`

## 5) Model Pipeline

**Features used (both apps):** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

**Python** (`src/app/main.py`):

Preprocess:
- Age median imputation
- Sex, Embarked most-frequent impute + one-hot
- Pclass, SibSp, Parch, Fare as numeric
  
Model: `LogisticRegression(max_iter=1000)`

Output: `pred_test_py.csv` with `PassengerId, Survived`

**R** (`src/rapp/main.R`):
  
Preprocess:
- Numeric (`Age, SibSp, Parch, Fare`) median imputation
- Sex, Embarked mode impute, and one-hot using train levels
  
Model: `glm(..., family = binomial())`

Output: `pred_test_r.csv` with `PassengerId, Survived`

## 6) Verify outputs
After running both:
```bash
src/data/
├─ train.csv
├─ test.csv
├─ pred_test_py.csv  # from Python container
└─ pred_test_r.csv   # from R container
```

## 7) Common commands
Rebuild after code/deps changes:
```bash
docker build -t titanic-ml .
docker build -t titanic-r -f src/rapp/Dockerfile .
```

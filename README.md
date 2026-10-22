# EduPredict: Student Performance Prediction System

A college-project upgrade of the original Streamlit app. Includes a premium navy dashboard, public-dataset ML pipeline, SQLite persistence, password hashing, student/admin roles, saved prediction history, and PDF/CSV exports.

## Dataset and attribution

- Dataset: **Student Performance**, by Paulo Cortez (2008), UCI Machine Learning Repository.
- Official page: https://archive.ics.uci.edu/dataset/320/student+performance
- License: Creative Commons Attribution 4.0 (CC BY 4.0).
- This app uses the mathematics file (`student-mat.csv`), which has 395 records and final grade `G3` on a 0–20 scale. The dataset describes students from two Portuguese secondary schools, so results may not generalize to other institutions or countries.
- Download mirror used by the app if the CSV is absent: https://huggingface.co/datasets/DElmazi/Student_Performance/resolve/main/student-mat.csv

## Requirements

Python 3.10 or 3.11 recommended.

## Run on Windows

1. Extract the ZIP.
2. Open the extracted `EduPredict_Advanced` folder in File Explorer.
3. Click the address bar, type `cmd`, and press Enter.
4. Run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

5. Open the local URL Streamlit prints, normally `http://localhost:8501`.

On macOS/Linux activate with `source .venv/bin/activate`.

## Initial administrator

On first run, the app creates:

- Username: `admin`
- Password: `Admin@12345`

Sign in and change this password from **Account** immediately. Do not expose a deployed demo with the default password.

Students can create their own accounts from the login screen. Passwords are stored as PBKDF2-SHA256 hashes, not plaintext.

## Model training and evaluation

The app downloads `student-mat.csv` on first model load if it is missing, then trains and compares Random Forest, Extra Trees, and HistGradientBoosting regressors. It selects the model with lowest MAE on an 80/20 train/test split (random state 42) and saves the fitted pipeline and held-out metrics.

You can explicitly train it from the project folder:

```bash
python -m models.train_model
```

Evaluation metrics (MAE, RMSE, R²) are computed only on the held-out test split. Do not compare results to published scores unless the split and features are the same. G1 and G2 are included and are strongly correlated with final grade G3. The UI flags this limitation; for earlier intervention, evaluate a separate model that excludes G1/G2.

## Database

SQLite database is created at `data/edupredict.db` on first run. It stores accounts and prediction history. Back up the database before deleting the folder. The bundled `.gitignore` excludes the database and trained model artifacts.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Main features

- Dark navy-blue responsive dashboard.
- Student registration and sign-in.
- Admin and student roles.
- Student prediction form using UCI academic features.
- SQLite records and prediction history.
- Model comparison and feature importance view for admins.
- PDF and CSV prediction report download.
- Input validation and test coverage.

## Limitations

This is an educational demonstration, not a production school information system. It does not include email verification, password reset, rate limiting, multi-factor authentication, or hardened deployment configuration. The UCI data is small and geographically limited. Predictions should not be used as the sole basis for educational decisions.

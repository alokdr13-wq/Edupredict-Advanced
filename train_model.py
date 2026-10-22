"""Train and evaluate on the public UCI Student Performance mathematics dataset."""
from pathlib import Path
import json, urllib.request
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
MODEL_DIR=ROOT/"models"
CSV=DATA/"student-mat.csv"
MODEL=MODEL_DIR/"model_bundle.joblib"
METRICS=MODEL_DIR/"metrics.json"
URL="https://huggingface.co/datasets/DElmazi/Student_Performance/resolve/main/student-mat.csv"
TARGET="G3"
FEATURES=["school","sex","age","address","studytime","failures","absences","G1","G2","schoolsup"]

def get_dataset():
    DATA.mkdir(exist_ok=True)
    if not CSV.exists():
        try:
            req=urllib.request.Request(URL,headers={"User-Agent":"EduPredict-College-Project/1.0"})
            with urllib.request.urlopen(req,timeout=25) as response:
                CSV.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(f"Could not download UCI dataset mirror. Check internet, or download student-mat.csv from {URL} into data/. Details: {exc}")
    df=pd.read_csv(CSV,sep=";")
    if TARGET not in df.columns or not set(FEATURES).issubset(df.columns):
        raise ValueError("Dataset columns do not match the expected UCI Student Performance schema.")
    return df

def make_pipeline(model):
    categorical=["school","sex","address","schoolsup"]
    numeric=[c for c in FEATURES if c not in categorical]
    pre=ColumnTransformer([
      ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
      ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
    ])
    return Pipeline([("preprocess",pre),("model",model)])

def train_and_evaluate():
    df=get_dataset()
    X=df[FEATURES].copy()
    y=pd.to_numeric(df[TARGET],errors="coerce")
    keep=y.notna()
    X,y=X.loc[keep],y.loc[keep]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    candidates={
      "Random Forest":RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=-1),
      "Extra Trees":ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=-1),
      "Hist Gradient Boosting":HistGradientBoostingRegressor(max_iter=180,learning_rate=.07,l2_regularization=1.0,random_state=42)
    }
    evaluated=[]
    for name,estimator in candidates.items():
        pipe=make_pipeline(estimator)
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test)
        evaluated.append({"name":name,"pipeline":pipe,"mae":float(mean_absolute_error(y_test,pred)),"rmse":float(np.sqrt(mean_squared_error(y_test,pred))),"r2":float(r2_score(y_test,pred))})
    best=min(evaluated,key=lambda m:m["mae"])
    # Refit the selected model on training split only, preserving held-out metrics.
    bundle={"pipeline":best["pipeline"],"features":FEATURES,"target":TARGET,"model_name":best["name"],"metrics":{"rows":int(len(df)),"train_rows":int(len(X_train)),"test_rows":int(len(X_test)),"mae":best["mae"],"rmse":best["rmse"],"r2":best["r2"],"comparison":[{k:v for k,v in item.items() if k not in ("pipeline",)} for item in evaluated],"dataset":"UCI Student Performance / mathematics","split":"80/20 train-test, random_state=42"}}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle,MODEL)
    METRICS.write_text(json.dumps(bundle["metrics"],indent=2),encoding="utf-8")
    return bundle

if __name__=="__main__":
    b=train_and_evaluate()
    print(json.dumps({"model":b["model_name"],"metrics":b["metrics"]},indent=2))

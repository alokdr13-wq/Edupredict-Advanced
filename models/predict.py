from pathlib import Path
import json
import pandas as pd
import joblib
from models.train_model import MODEL, METRICS, FEATURES, train_and_evaluate

def load_model_bundle():
    if not MODEL.exists():
        return train_and_evaluate()
    return joblib.load(MODEL)

def predict_grade(bundle, row):
    clean={key:row.get(key) for key in bundle["features"]}
    prediction=float(bundle["pipeline"].predict(pd.DataFrame([clean]))[0])
    return round(min(20.0,max(0.0,prediction)),2)

def get_model_metrics(bundle):
    metrics=bundle.get("metrics")
    if not metrics and METRICS.exists():
        metrics=json.loads(METRICS.read_text(encoding="utf-8"))
    if not metrics:
        return {"rows":0,"mae":0.0,"rmse":0.0,"r2":0.0}
    return metrics

def feature_importance(bundle):
    model=bundle["pipeline"].named_steps["model"]
    prep=bundle["pipeline"].named_steps["preprocess"]
    try:
        names=prep.get_feature_names_out()
        values=model.feature_importances_
        pairs=[]
        for n,v in zip(names,values):
            base=n.split("__",1)[-1]
            pairs.append((base,float(v)))
        grouped={}
        for n,v in pairs:
            key=n.split("_")[0] if n not in bundle["features"] else n
            grouped[key]=grouped.get(key,0)+v
        return pd.DataFrame(sorted(grouped.items(),key=lambda x:x[1],reverse=True),columns=["Feature","Importance"]).head(12)
    except Exception:
        return pd.DataFrame({"Feature":bundle["features"],"Importance":[0]*len(bundle["features"])})

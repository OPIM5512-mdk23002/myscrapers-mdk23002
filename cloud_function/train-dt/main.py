# Tuned GradientBoosting model: train on all data < today (local TZ); hold out today
# Saves predictions, metrics, permutation importance, and PDP plots to GCS
# HTTP entrypoint: train_dt_http

import os, io, json, logging, traceback, re
import numpy as np
import pandas as pd
#setting matplotlib non interactive backend for cloud function environment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "structured/preds")            # e.g., "structured/preds"
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")      # split by local day
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
#saving dicts to GCS as JSON files
def _write_json_to_gcs(client, bucket, key, data):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
#saving matplotlib figures to GCS as PNGs
def _write_png_to_gcs(client, bucket, key, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi = 150, bbox_inches="tight")
    buf.seek(0)
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(buf.read(), content_type="image/png")
    plt.close(fig)

def _clean_numeric(s: pd.Series) -> pd.Series:
    # Strip $, commas, spaces; keep digits and dot
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def run_once(dry_run= False):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date
    df = df[df["date_local"].notna()].copy()
    # --- Clean numerics BEFORE counting/dropping ---
    orig_rows = len(df)
    df["price_num"]   = _clean_numeric(df["price"])
    df["year_num"]    = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])
    df["engine_cylinders"] = _clean_numeric(df["engine_cylinders"])  # new llm field as numeric

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | with valid numeric price=%d", orig_rows, valid_price_rows)

    counts = df["date_local"].value_counts().sort_index()
    logging.info("Recent date counts (local): %s", json.dumps({str(k): int(v) for k, v in counts.tail(8).items()}))

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two distinct dates", "dates": [str(d) for d in unique_dates]}

    today_local = unique_dates[-1]
    train_df   = df[df["date_local"] <  today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    train_df = train_df[train_df["price_num"].notna()]
    dropped_for_target = int((df["date_local"] < today_local).sum()) - int(len(train_df))
    logging.info("Train rows after target clean: %d (dropped_for_target=%d)", len(train_df), dropped_for_target)
    logging.info("Holdout rows today (%s): %d", today_local, len(holdout_df))

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    #Features: core + LLM-extracted fields
    target = "price_num"
    cat_cols = ["make", "model", "transmission", "drivetrain", "fuel_type", "condition", "color", "body_type", "title_status"]
    num_cols = ["year_num", "mileage_num", "engine_cylinders"]  # included engine_cylinders as numeric
    feats = cat_cols + num_cols

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )
    #Model with hyperparameter tuning through GridSearchCV
    base_model = GradientBoostingRegressor(random_state=23002,)
    pipe = Pipeline([("pre", pre), ("model", base_model)])

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [5, 7],
        "model__learning_rate": [0.05, 0.1],
        "model__min_samples_leaf": [5, 10],
    }

    search = GridSearchCV(
        pipe, 
        param_grid, 
        scoring="neg_mean_absolute_error", 
        cv=3, 
        n_jobs=-1, 
        refit=True, 
        verbose=1,
        )

    X_train = train_df[feats]
    y_train = train_df[target]
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    best_params = search.best_params_
    best_cv_mae = float(-search.best_score_)

    logging.info("Best params: %s", best_params)
    logging.info("Best CV MAE: %.2f", best_cv_mae)

    # ---- Predict/evaluate on today's holdout (now includes actual price fields) ----
    mae_today = rmse_today = mape_today = bias_today = None
    preds_df = pd.DataFrame()

    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat = best_pipe.predict(X_h)

        cols_out = ["post_id", "scraped_at", "make", "model", "year", "mileage", "price", "transmission", "drivetrain", "fuel_type", "engine_cylinders", 
                    "condition", "color", "body_type", "title_status"]  # core + LLM fields for output
        preds_df = holdout_df[cols_out].copy()
        preds_df["actual_price"] = holdout_df["price_num"]       # cleaned numeric truth
        preds_df["pred_price"]   = np.round(y_hat, 2)

        y_true = holdout_df["price_num"]
        mask = y_true.notna()
        if mask.any():
            yt = y_true[mask]
            yp = y_hat[mask]
            mae_today  = float(mean_absolute_error(yt, yp))
            rmse_today = float(np.sqrt(mean_squared_error(yt, yp)))
            mape_today = float(np.mean(np.abs((yt - yp) / yt)) * 100)
            bias_today = float(np.mean(yp - yt))

    # --- Output path: HOURLY folder structure ---
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    run_folder = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}"

    if not dry_run and len(preds_df) > 0:
      
      #Prediction output CSV
      preds_key = f"{run_folder}/preds.csv"
      _write_csv_to_gcs(client, GCS_BUCKET, preds_key, preds_df)
      logging.info("Saved predictions to gs://%s/%s", GCS_BUCKET, preds_key)
      #Metrics JSON for trending notebook  
      metrics = {
          "run_ts": now_utc.isoformat(),
          "today_local": str(today_local),
          "train_rows": int(len(train_df)),
          "holdout_rows": int(len(holdout_df)),
          "mae_today": mae_today,
          "rmse_today": rmse_today,
          "mape_today": mape_today,
          "bias_today": bias_today,
          "best_cv_mae": best_cv_mae,
          "best_params": best_params,
      }
      metrics_key = f"{run_folder}/metrics.json"
      _write_json_to_gcs(client, GCS_BUCKET, metrics_key, metrics)
      logging.info("Saved metrics to gs://%s/%s", GCS_BUCKET, metrics_key)

      #Permutation importance JSON
      if mask.any():
          perm_result = permutation_importance(
              best_pipe, X_h[mask], yt,
              n_repeats=30, random_state=23002,
              scoring="neg_mean_absolute_error")
          
          feature_names = feats
          perm_df = pd.DataFrame({
              "feature": feature_names,
              "importance_mean": perm_result.importances_mean,}).sort_values("importance_mean", ascending=False)
          
          #Plotting permutation importance in ascending order for better visualization
          perm_plot = perm_df.sort_values("importance_mean", ascending=True)
          fig_pi, ax_pi = plt.subplots(figsize=(8, 5))
          ax_pi.barh(perm_plot["feature"], perm_plot["importance_mean"], color="blue")
          ax_pi.set_xlabel("Mean MAE Increase")
          ax_pi.set_title("Permutation Importance")


          perm_data_key = f"{run_folder}/permutation_importance.json"
          _write_json_to_gcs(client, GCS_BUCKET, perm_data_key, perm_df.to_dict(orient="records"))
          logging.info("Saved permutation importance JSON to gs://%s/%s", GCS_BUCKET, perm_data_key)

          pi_key = f"{run_folder}/permutation_importance.png"
          _write_png_to_gcs(client, GCS_BUCKET, pi_key, fig_pi)
          logging.info("Saved permutation importance plot to gs://%s/%s", GCS_BUCKET, pi_key)

          #Partial Dependence Plots for top 3 features
          #Only include features that were non-null in training which the imputer could actually learn from
          #and that have data in holdout for PDP to compute on
          valid_feats = [f for f in perm_df["feature"]
                    if X_train[f].notna().any() and X_h[f].notna().any()]
          top_3 = valid_feats[:3]

          # Fill NaN in categorical columns so PDP can sort unique values
          X_h_pdp = X_h.copy()
          for col in cat_cols:
              if col in X_h_pdp.columns:
                  X_h_pdp[col] = X_h_pdp[col].fillna("unknown")
          
          fig_pdp, axes_pdp = plt.subplots(1, 3, figsize=(15, 5))
          PartialDependenceDisplay.from_estimator(
              best_pipe, X_h_pdp, features=top_3,
              feature_names=feats,
              ax=axes_pdp,
              kind="average",
              categorical_features=cat_cols,
          )
          fig_pdp.suptitle(f"Partial Dependence — Top 3 Features — {today_local}", fontsize=14)
          plt.tight_layout()

          pdp_key = f"{run_folder}/pdp_top3.png"
          _write_png_to_gcs(client, GCS_BUCKET, pdp_key, fig_pdp)
          logging.info("Wrote PDP to gs://%s/%s", GCS_BUCKET, pdp_key)

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_price_rows": valid_price_rows,
        "mae_today": mae_today,
        "rmse_today": rmse_today,
        "mape_today": mape_today,
        "bias_today": bias_today,
        "best_cv_mae": best_cv_mae,
        "best_params": best_params,
        "output_folder": run_folder,
        "dry_run": dry_run,
        "timezone": TIMEZONE,
    }

def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False)),
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})

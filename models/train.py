# models/train.py

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold, RepeatedKFold, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, make_scorer
import time
import os
from utils.logger import get_logger

# define rmse natively
def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

import numpy as np
import pandas as pd


LOGGER = get_logger(__name__)


def _safe_call(cb, *args, context: str):
    if not callable(cb):
        return
    try:
        cb(*args)
    except Exception:
        LOGGER.exception("Non-fatal callback failure (%s)", context)

from config import RSTATE, CV_MODE, CV_FOLDS, CV_REPEATS, NESTED_OUTER_FOLDS, NESTED_INNER_FOLDS
try:
    from config import CV_STRATIFY
except Exception:
    CV_STRATIFY = 'none'
from config.selections import get_selected_models


def train_and_evaluate(
    X, y, preprocessor, model_names=None, progress_callback=None,
    cv_mode='kfold', cv_folds=5, log_callback=None, model_status_callback=None
):
    model_names = get_selected_models(model_names)

    all_models = {
        "LinearRegression": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=RSTATE, n_jobs=-1),
        "HistGB": HistGradientBoostingRegressor(
            random_state=RSTATE, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=RSTATE, validation_fraction=0.1, n_iter_no_change=10
        ),
        "Lasso": LassoCV(alphas=np.logspace(-3, 3, 13), random_state=RSTATE, cv=5),
        "ElasticNet": ElasticNetCV(alphas=np.logspace(-3, 3, 13), l1_ratio=[0.1, 0.5, 0.9], random_state=RSTATE, cv=5),
        "SVR": SVR(),
        "KNeighborsRegressor": KNeighborsRegressor(),
    }

    if XGB_OK:
        all_models["XGBoost"] = XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method='hist', random_state=RSTATE, n_jobs=-1
        )

    scoring = {
        "R2": make_scorer(r2_score),
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "RMSE": make_scorer(rmse, greater_is_better=False),
    }

    rows, fitted = [], {}
    # Optional y-decile stratification for regression
    y_strata = None
    try:
        if CV_STRATIFY and str(CV_STRATIFY).lower() == 'deciles':
            y_series = pd.Series(y)
            # use rank to avoid duplicate edges, then qcut
            y_strata = pd.qcut(y_series.rank(method='first'), q=10, labels=False, duplicates='drop')
    except Exception:
        y_strata = None
    total = len(model_names)
    for idx, name in enumerate(model_names):
        if name not in all_models:
            continue
        pipe = Pipeline([("prep", preprocessor), ("model", all_models[name])])

        oof_pred = None

        if callable(log_callback):
            _safe_call(log_callback, f"[Start] {name}", context="log_callback:start")
        if callable(model_status_callback):
            _safe_call(model_status_callback, name, "start", context="model_status_callback:start")

        # — run either repeated/k-fold/nested CV or simple hold-out
        mode = cv_mode or CV_MODE
        per_split = {}
        if mode == 'kfold':
            if y_strata is not None:
                skf = StratifiedKFold(n_splits=cv_folds or CV_FOLDS, shuffle=True, random_state=RSTATE)
                cv = skf.split(X, y_strata)
            else:
                cv = KFold(n_splits=cv_folds or CV_FOLDS, shuffle=True, random_state=RSTATE)
            cvres = cross_validate(
                pipe, X, y, cv=cv, scoring=scoring,
                return_train_score=False
            , n_jobs=-1)
            pipe.fit(X, y)
            final_pipe = pipe
            r2_mean = cvres["test_R2"].mean()
            mae_mean = -cvres["test_MAE"].mean()
            rmse_mean = -cvres["test_RMSE"].mean()
            train_time = cvres["fit_time"].mean()
            per_split = {
                "R2": list(cvres["test_R2"]),
                "MAE": list(-cvres["test_MAE"]),
                "RMSE": list(-cvres["test_RMSE"]),
            }
        elif mode == 'repeated':
            if y_strata is not None:
                from sklearn.model_selection import RepeatedStratifiedKFold
                cv = RepeatedStratifiedKFold(n_splits=cv_folds or CV_FOLDS, n_repeats=CV_REPEATS, random_state=RSTATE).split(X, y_strata)
            else:
                cv = RepeatedKFold(n_splits=cv_folds or CV_FOLDS, n_repeats=CV_REPEATS, random_state=RSTATE)
            cvres = cross_validate(
                pipe, X, y, cv=cv, scoring=scoring, return_train_score=False
            , n_jobs=-1)
            pipe.fit(X, y)
            final_pipe = pipe
            r2_mean = cvres["test_R2"].mean()
            mae_mean = -cvres["test_MAE"].mean()
            rmse_mean = -cvres["test_RMSE"].mean()
            train_time = cvres["fit_time"].mean()
            per_split = {
                "R2": list(cvres["test_R2"]),
                "MAE": list(-cvres["test_MAE"]),
                "RMSE": list(-cvres["test_RMSE"]),
            }
        elif mode == 'nested':
            # Minimal nested CV with light grids to control runtime
            param_grid = {}
            if name == "RidgeCV":
                param_grid = {"model__alpha": list(np.logspace(-3, 3, 13))}
                from sklearn.linear_model import Ridge
                pipe = Pipeline([("prep", preprocessor), ("model", Ridge())])
            elif name == "RandomForest":
                rf_grid_mode = os.environ.get("NESTED_RF_GRID", "light").lower()
                if rf_grid_mode == "ultra_light":
                    param_grid = {"model__n_estimators": [200], "model__max_depth": [None, 10]}
                elif rf_grid_mode == "full":
                    param_grid = {"model__n_estimators": [300, 500], "model__max_depth": [None, 8, 16]}
                else:
                    param_grid = {"model__n_estimators": [200, 400], "model__max_depth": [None, 10]}
            elif name == "HistGB":
                param_grid = {"model__max_depth": [None, 6, 12], "model__learning_rate": [0.05, 0.1]}
            elif name == "ElasticNet":
                param_grid = {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}
            elif name == "SVR":
                param_grid = {"model__C": [1.0, 10.0, 100.0], "model__gamma": ["scale", 0.1, 0.01]}
            elif name == "KNeighborsRegressor":
                param_grid = {"model__n_neighbors": [3, 5, 11], "model__weights": ["uniform", "distance"]}
            elif name == "XGBoost" and XGB_OK:
                param_grid = {"model__n_estimators": [200, 400], "model__learning_rate": [0.05, 0.1], "model__max_depth": [4, 6]}

            outer_folds = int(cv_folds or NESTED_OUTER_FOLDS)
            inner_folds = int(NESTED_INNER_FOLDS)

            if y_strata is not None:
                skf_outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RSTATE)
                outer = list(skf_outer.split(X, y_strata))
            else:
                outer = KFold(n_splits=outer_folds, shuffle=True, random_state=RSTATE)
            
            # GridSearchCV does cv.split(X, y), which breaks StratifiedKFold for continuous y, so we use KFold inner
            inner = KFold(n_splits=inner_folds, shuffle=True, random_state=RSTATE)

            if param_grid:
                gs = GridSearchCV(pipe, param_grid=param_grid, cv=inner,
                                  scoring={"R2": scoring["R2"], "MAE": scoring["MAE"], "RMSE": scoring["RMSE"]},
                                  refit="R2", n_jobs=-1, pre_dispatch="2*n_jobs")
                cvres = cross_validate(gs, X, y, cv=outer,
                                       scoring=scoring, return_estimator=True, return_train_score=False, n_jobs=1)
                
                gs_final = GridSearchCV(pipe, param_grid=param_grid, cv=inner, scoring=scoring["R2"], refit=True, n_jobs=-1)
                gs_final.fit(X, y)
                final_pipe = gs_final.best_estimator_
            else:
                cvres = cross_validate(pipe, X, y, cv=outer,
                                       scoring=scoring, return_estimator=True, return_train_score=False, n_jobs=-1, pre_dispatch="2*n_jobs")
                pipe.fit(X, y)
                final_pipe = pipe

            # Out-of-fold predictions aligned to outer CV (publication-safe diagnostics for nested mode).
            try:
                ests = cvres.get("estimator")
                if ests is not None:
                    outer_splits = outer if isinstance(outer, list) else list(outer.split(X, y))
                    y_oof = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)
                    for (tr_idx, te_idx), est in zip(outer_splits, ests):
                        X_te = X.iloc[te_idx] if hasattr(X, "iloc") else X[te_idx]
                        pred = np.asarray(est.predict(X_te), dtype=float).reshape(-1)
                        y_oof[np.asarray(te_idx, dtype=int)] = pred
                    oof_pred = y_oof
            except Exception:
                oof_pred = None

            r2_mean = cvres["test_R2"].mean()
            mae_mean = -cvres["test_MAE"].mean()
            rmse_mean = -cvres["test_RMSE"].mean()
            train_time = (cvres.get("fit_time", np.array([0])).mean())
            per_split = {
                "R2": list(cvres["test_R2"]),
                "MAE": list(-cvres["test_MAE"]),
                "RMSE": list(-cvres["test_RMSE"]),
            }
        else:  # hold-out
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=RSTATE
            )
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            train_time = time.time() - t0
            y_pred = pipe.predict(X_te)
            r2_mean = r2_score(y_te, y_pred)
            mae_mean = mean_absolute_error(y_te, y_pred)
            rmse_mean = rmse(y_te, y_pred)
            
            pipe.fit(X, y)
            final_pipe = pipe

        y_hat = final_pipe.predict(X)
        # compute training statistics
        r2_train = r2_score(y, y_hat)
        mae_train = mean_absolute_error(y, y_hat)
        rmse_train = rmse(y, y_hat)
        rows.append({
            "model":       name,
            "R2_CV":       float(r2_mean),
            "MAE_CV":      float(mae_mean),
            "RMSE_CV":     float(rmse_mean),
            "R2_train":    float(r2_train),
            "MAE_train":   float(mae_train),
            "RMSE_train":  float(rmse_train),
            "TrainingTime": float(train_time)
        })
        fitted[name] = {
            "pipe": final_pipe,
            "holdout": (X, y, y_hat),
            "cv_mode": mode,
            "cv_scores": per_split,
            "oof_pred": oof_pred,
        }
        if progress_callback is not None:
            _safe_call(progress_callback, idx + 1, total, context="progress_callback")

        if callable(log_callback):
            _safe_call(
                log_callback,
                f"[Done] {name} | R2_CV={r2_mean:.4f} RMSE_CV={rmse_mean:.4f}",
                context="log_callback:done",
            )
        if callable(model_status_callback):
            _safe_call(model_status_callback, name, "done", context="model_status_callback:done")

    metrics_df = pd.DataFrame(rows).sort_values("R2_CV", ascending=False)
    return metrics_df, fitted

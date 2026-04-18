import re

with open('models/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to rewrite the evaluation block exactly so that it handles hold-out correctly,
# and kfold/repeated/nested actually run correctly without clobbering each other.

replacement = """        mode = cv_mode or CV_MODE
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

            if y_strata is not None:
                skf_outer = StratifiedKFold(n_splits=NESTED_OUTER_FOLDS, shuffle=True, random_state=RSTATE)
                outer = skf_outer.split(X, y_strata)
            else:
                outer = KFold(n_splits=NESTED_OUTER_FOLDS, shuffle=True, random_state=RSTATE)
            if y_strata is not None:
                inner = StratifiedKFold(n_splits=NESTED_INNER_FOLDS, shuffle=True, random_state=RSTATE)
            else:
                inner = KFold(n_splits=NESTED_INNER_FOLDS, shuffle=True, random_state=RSTATE)

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

        y_hat = final_pipe.predict(X)"""

pattern = r"        mode = cv_mode or CV_MODE\n        per_split = None.*?        y_hat = final_pipe\.predict\(X\)"
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('models/train.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

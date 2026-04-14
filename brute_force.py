from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from skimage.feature import hog

n_jobs = -1  # Use all available CPU cores for parallel processing
verbose = 2  # Set verbose level for GridSearchCV to get detailed output during the search process

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values (X = images, y = distances)
    X, y = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(X)} samples.")

    # Split the dataset into training and testing sets of Train_images
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLPRegressor 
    MLPr = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('model', MLPRegressor(hidden_layer_sizes=(100, 30, 30, 50), max_iter=3000, random_state=42))
    ])

    param_grid = {
        "pca__n_components": [50, 100, 200, 300, 400],
        "model__hidden_layer_sizes": [(128, 64),(256, 256, 256), (256, 64, 64, 128), (256, 128, 128, 64)],
        "model__alpha": [0.05, 0.1, 0.2, 0.5]
        }

    # Grid Search 
    grid_mlpr = GridSearchCV(MLPr, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error',n_jobs=n_jobs, verbose=verbose)
    grid_mlpr.fit(X_train, y_train)

    best_model_mlpr = grid_mlpr.best_estimator_

    # Evaluation of the model
    y_pred = best_model_mlpr.predict(X_test)
    mae_mlp = mean_absolute_error(y_test, y_pred)
    r2_mlp = r2_score(y_test, y_pred)
    print(f"MLPRegressor: Mean Absolute Error: {mae_mlp}, MLPRegressor R2 Score: {r2_mlp}")
    print("Best MLPRegressor model:", grid_mlpr.best_estimator_)


    # 2.
    # Random Forest Regressor 
    RFR = Pipeline([
        ('model', RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [30 ,50, 100, 200],
        "model__max_depth": [None, 10, 20, 30]
        }

    # Grid Search 
    grid_rfr = GridSearchCV(RFR, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error',n_jobs=n_jobs, verbose=verbose)
    grid_rfr.fit(X_train, y_train)

    best_model_rfr = grid_rfr.best_estimator_

    # Evaluation of the model
    y_pred = best_model_rfr.predict(X_test)
    mae_rfr = mean_absolute_error(y_test, y_pred)
    r2_rfr = r2_score(y_test, y_pred)
    print(f"RandomForestRegressor: Mean Absolute Error: {mae_rfr}, RandomForestRegressor R2 Score: {r2_rfr}")
    print("Best RandomForestRegressor model:", grid_rfr.best_estimator_)

    # 3
    # extra tree regressor
    ETR = Pipeline([
        ('model', ExtraTreesRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [30 ,50, 100, 200],
        "model__max_depth": [None, 10, 20, 30]
    }

    # Grid Search
    grid_etr = GridSearchCV(ETR, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error',n_jobs=n_jobs, verbose=verbose)
    grid_etr.fit(X_train, y_train)

    best_model_etr = grid_etr.best_estimator_

    # Evaluation of the model
    y_pred = best_model_etr.predict(X_test)
    mae_etr = mean_absolute_error(y_test, y_pred)
    r2_etr = r2_score(y_test, y_pred)
    print(f"ExtraTreeRegressor: Mean Absolute Error: {mae_etr}, ExtraTreeRegressor R2 Score: {r2_etr}")
    print("Best ExtraTreeRegressor model:", grid_etr.best_estimator_)


    # 4
    # Gradient Boosting Regressor
    GBR = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [30, 50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2]
    }

    # Grid Search
    grid_gbr = GridSearchCV(GBR, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error',n_jobs=n_jobs, verbose=verbose)
    grid_gbr.fit(X_train, y_train)

    best_model_gbr = grid_gbr.best_estimator_

    # Evaluation of the model
    y_pred = best_model_gbr.predict(X_test)
    mae_gb = mean_absolute_error(y_test, y_pred)
    r2_gb = r2_score(y_test, y_pred)
    print(f"GradientBoostingRegressor: Mean Absolute Error: {mae_gb}, GradientBoostingRegressor R2 Score: {r2_gb}")
    print("Best GradientBoostingRegressor model:", grid_gbr.best_estimator_)


    # 5
    # HistGradientBoostingRegressor
    HGBR = Pipeline([
        ("pca", PCA()),
        ("model", HistGradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        "pca__n_components": [50, 100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_iter": [200, 400,800],
    }

    # Grid Search
    grid_hgbr = GridSearchCV(HGBR, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=verbose)
    grid_hgbr.fit(X_train, y_train)

    best_model_hgbr = grid_hgbr.best_estimator_

    # Evaluation of the model
    y_pred = best_model_hgbr.predict(X_test)
    mae_hgbr = mean_absolute_error(y_test, y_pred)
    r2_hgbr = r2_score(y_test, y_pred)
    print(f"HistGradientBoostingRegressor: Mean Absolute Error: {mae_hgbr}, HistGradientBoostingRegressor R2 Score: {r2_hgbr}")
    print("Best HistGradientBoostingRegressor model:", grid_hgbr.best_estimator_)


   # 6
   # KNeighbors Regressor
    KNN = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('model', KNeighborsRegressor())
    ])

    param_grid = [                              # Mit und ohne PCA testen
    {
        "pca": [PCA()],
        "pca__n_components": [20, 50, 100, 150],
        "model__n_neighbors": range(1, 30, 2),
        "model__weights": ["uniform", "distance"],
    },
    {
        "pca": ["passthrough"],
        "model__n_neighbors": range(1, 30, 2),
        "model__weights": ["uniform", "distance"],
    },
]

    # Grid Search
    grid_knn = GridSearchCV(KNN, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=verbose)
    grid_knn.fit(X_train, y_train)

    best_model_knn = grid_knn.best_estimator_

    # Evaluation of the model
    y_pred = best_model_knn.predict(X_test)
    mae_knn = mean_absolute_error(y_test, y_pred)
    r2_knn = r2_score(y_test, y_pred)
    print(f"KNeighborsRegressor: Mean Absolute Error: {mae_knn}, KNeighborsRegressor R2 Score: {r2_knn}")
    print("Best KNeighborsRegressor model:", grid_knn.best_estimator_)


    #7
    # ridge regression
    RidgeReg = Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', PCA()),
        ('model', Ridge(random_state=42))    
    ])

    param_grid = {
        "pca__n_components": [50, 100, 200, 300, 400],
        "model__alpha": [0.1, 1.0, 10.0, 100.0]
    }

    # Grid Search
    grid_ridge = GridSearchCV(RidgeReg, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=verbose)
    grid_ridge.fit(X_train, y_train)

    best_model_ridge = grid_ridge.best_estimator_

    # Evaluation of the model
    y_pred = best_model_ridge.predict(X_test)
    mae_ridge = mean_absolute_error(y_test, y_pred)
    r2_ridge = r2_score(y_test, y_pred)
    print(f"Ridge Regression: Mean Absolute Error: {mae_ridge}, Ridge R2 Score: {r2_ridge}")
    print("Best Ridge Regression model:", grid_ridge.best_estimator_)

    """"
    # 8
    # Polynomial Regression
    PolyReg = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('model', LinearRegression())
    ])

    param_grid = {
        "poly__degree": [2, 3, 4],
        "poly__interaction_only": [False, True],
        "poly__include_bias": [False, True]
    }

    # Grid Search
    grid_poly = GridSearchCV(PolyReg, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=verbose)
    grid_poly.fit(X_train, y_train)

    best_model_poly = grid_poly.best_estimator_

    # Evaluation of the model
    y_pred = best_model_poly.predict(X_test)
    mae_poly = mean_absolute_error(y_test, y_pred)
    r2_poly = r2_score(y_test, y_pred)
    print(f"Polynomial Regression: Mean Absolute Error: {mae_poly}, Polynomial Regression R2 Score: {r2_poly}")
    print("Best Polynomial Regression model:", grid_poly.best_estimator_)
    """

    # 9
    # Gaussian Process Regressor
    GPR = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('model', GaussianProcessRegressor(random_state=42))
    ])

    param_grid = {
        "pca__n_components": [50, 100, 200, 300],
        "model__alpha": [1e-10, 1e-5, 1e-2, 0.1],
    }

    # Grid Search
    grid_gpr = GridSearchCV(GPR, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=verbose)
    grid_gpr.fit(X_train, y_train)

    best_model_gpr = grid_gpr.best_estimator_

    # Evaluation of the model
    y_pred = best_model_gpr.predict(X_test)
    mae_gpr = mean_absolute_error(y_test, y_pred)
    r2_gpr = r2_score(y_test, y_pred)
    print(f"GaussianProcessRegressor: Mean Absolute Error: {mae_gpr}, GaussianProcessRegressor R2 Score: {r2_gpr}")
    print("Best GaussianProcessRegressor model:", grid_gpr.best_estimator_)


    # 10
    # PLS Regression   
    PLSR = Pipeline([
        ('scaler', StandardScaler()),
        ('model', PLSRegression())
    ])

    param_grid = {
        "model__n_components": [2, 5, 10, 20, 50]
    }

    # Grid Search
    grid_plsr = GridSearchCV(PLSR, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=verbose)
    grid_plsr.fit(X_train, y_train)

    best_model_plsr = grid_plsr.best_estimator_

    # Evaluation of the model
    y_pred = best_model_plsr.predict(X_test)
    mae_plsr = mean_absolute_error(y_test, y_pred)
    r2_plsr = r2_score(y_test, y_pred)
    print(f"PLSRegression: Mean Absolute Error: {mae_plsr}, PLSRegression R2 Score: {r2_plsr}")
    print("Best PLSRegression model:", grid_plsr.best_estimator_)


    #11
    # ElasticNet Regression
    ElasticNetReg = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('model', ElasticNet(random_state=42, max_iter=7000))
    ])

    param_grid = {
        "pca__n_components": [50, 100, 200, 300, 400],
        "model__alpha": [0.1, 0.5, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.5, 0.9]
    }

    # Grid Search
    grid_enet = GridSearchCV(ElasticNetReg, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=verbose)
    grid_enet.fit(X_train, y_train)
    best_model_enet = grid_enet.best_estimator_

    # Evaluation of the model
    y_pred = best_model_enet.predict(X_test)
    mae_enet = mean_absolute_error(y_test, y_pred) 
    r2_enet = r2_score(y_test, y_pred)
    print(f"ElasticNet Regression: Mean Absolute Error: {mae_enet}, ElasticNet Regression R2 Score: {r2_enet}")
    print("Best ElasticNet Regression model:", grid_enet.best_estimator_)



    results = [
        {"name": "MLPRegressor", "mae": mae_mlp, "r2": r2_mlp},
        {"name": "RandomForestRegressor", "mae": mae_rfr, "r2": r2_rfr},
        {"name": "ExtraTreeRegressor", "mae": mae_etr, "r2": r2_etr},
        {"name": "GradientBoostingRegressor", "mae": mae_gb, "r2": r2_gb},
        {"name": "HistGradientBoostingRegressor", "mae": mae_hgbr, "r2": r2_hgbr},
        {"name": "KNeighborsRegressor", "mae": mae_knn, "r2": r2_knn},
        {"name": "Ridge Regression", "mae": mae_ridge, "r2": r2_ridge},
        {"name": "Polynomial Regression", "mae": mae_poly, "r2": r2_poly},
        {"name": "GaussianProcessRegressor", "mae": mae_gpr, "r2": r2_gpr},
        {"name": "PLSRegression", "mae": mae_plsr, "r2": r2_plsr},
        {"name": "ElasticNet Regression", "mae": mae_enet, "r2": r2_enet},
    ]

    results_sorted = sorted(results, key=lambda result: result["mae"], reverse=True)

    for result in results_sorted:
        print(f"{result['name']}: Mean Absolute Error: {result['mae']}, R2 Score: {result['r2']}")    
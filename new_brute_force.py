from utils import load_config, load_dataset
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score


n_jobs = -1  # Use all available CPU cores for parallel processing
verbose = 2  # Set verbose level for GridSearchCV to get detailed output during the search process
cv = KFold(n_splits=5, shuffle=True, random_state=42) # Define KFold cross-validator with 5 splits, shuffling, and a fixed random state for reproducibility

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
    grid_mlpr = GridSearchCV(MLPr, param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error',n_jobs=n_jobs, verbose=verbose)
    grid_mlpr.fit(X_train, y_train)

    best_model_mlpr = grid_mlpr.best_estimator_

    # Evaluation of the model
    y_pred = best_model_mlpr.predict(X_test)
    mae_mlp = mean_absolute_error(y_test, y_pred)
    r2_mlp = r2_score(y_test, y_pred)
    print(f"MLPRegressor: Mean Absolute Error: {mae_mlp}, MLPRegressor R2 Score: {r2_mlp}")
    print("Best MLPRegressor model:", grid_mlpr.best_estimator_)


    # extra tree regressor
    ETR = Pipeline([
        ('model', ExtraTreesRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [30 ,50, 100, 200],
        "model__max_depth": [None, 10, 20, 30]
    }

    # Grid Search
    grid_etr = GridSearchCV(ETR, param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error',n_jobs=n_jobs, verbose=verbose)
    grid_etr.fit(X_train, y_train)

    best_model_etr = grid_etr.best_estimator_

    # Evaluation of the model
    y_pred = best_model_etr.predict(X_test)
    mae_etr = mean_absolute_error(y_test, y_pred)
    r2_etr = r2_score(y_test, y_pred)
    print(f"ExtraTreeRegressor: Mean Absolute Error: {mae_etr}, ExtraTreeRegressor R2 Score: {r2_etr}")
    print("Best ExtraTreeRegressor model:", grid_etr.best_estimator_)


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
        "model__n_neighbors": range(1, 52, 2),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    {
        "pca": ["passthrough"],
        "model__n_neighbors": range(1, 52, 2),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
]

    # Grid Search
    grid_knn = GridSearchCV(KNN, param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=verbose)
    grid_knn.fit(X_train, y_train)

    best_model_knn = grid_knn.best_estimator_

    # Evaluation of the model
    y_pred = best_model_knn.predict(X_test)
    mae_knn = mean_absolute_error(y_test, y_pred)
    r2_knn = r2_score(y_test, y_pred)
    print(f"KNeighborsRegressor: Mean Absolute Error: {mae_knn}, KNeighborsRegressor R2 Score: {r2_knn}")
    print("Best KNeighborsRegressor model:", grid_knn.best_estimator_)

    results = [
        {"name": "MLPRegressor", "mae": mae_mlp, "r2": r2_mlp},
        {"name": "ExtraTreeRegressor", "mae": mae_etr, "r2": r2_etr},
        {"name": "KNeighborsRegressor", "mae": mae_knn, "r2": r2_knn}
    ]

    results_sorted = sorted(results, key=lambda result: result["mae"])

    print("\nSummary of Results:")
    for result in results_sorted:
        print(f"{result['name']}: Mean Absolute Error: {result['mae']}, R2 Score: {result['r2']}")
        

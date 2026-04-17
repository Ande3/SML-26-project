from numpy.testing import verbose

from utils import load_config, load_dataset, load_test_dataset, print_results, save_results

# sklearn imports...
# SVRs are not allowed in this project.

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score 

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values (X = images, y = distances)
    X, y = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(X)} samples.")

    # Split the dataset into training and testing sets of Train_images
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 
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
    grid_knn = GridSearchCV(KNN, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
    grid_knn.fit(X_train, y_train)

    best_model_knn = grid_knn.best_estimator_

    # Evaluation of the model
    y_pred = best_model_knn.predict(X_test)
    mae_knn = mean_absolute_error(y_test, y_pred)
    r2_knn = r2_score(y_test, y_pred)
    print(f"KNeighborsRegressor: Mean Absolute Error: {mae_knn}, KNeighborsRegressor R2 Score: {r2_knn}")
    print("Best KNeighborsRegressor model:", grid_knn.best_estimator_)

    """ # Train on whole Dataset and make predictions on test_images

    best_model.fit(X, y)
    X_final = load_test_dataset(config)
    y_final = best_model.predict(X_final)

    save_results(y_final)
    """
    



    

   
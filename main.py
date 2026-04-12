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
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("model", MLPRegressor(random_state=42, max_iter=3000))
    ])

    param_grid = {
        "pca__n_components": [5, 10, 30, 50, 100],
        "model__hidden_layer_sizes": [(64, 64 ,64, 64),(128, 128, 128), (256, 256, 256),],
        "model__alpha": [0.05, 0.1, 0.2, 0.5]
    }

    # Grid Search 
    grid = GridSearchCV(model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error',n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Evaluation of the model
    y_pred = best_model.predict(X_test)
    print_results(y_test, y_pred)
    print("Best model:", grid.best_estimator_)

    """ # Train on whole Dataset and make predictions on test_images

    best_model.fit(X, y)
    X_final = load_test_dataset(config)
    y_final = best_model.predict(X_final)

    save_results(y_final)
    """
    



    

   
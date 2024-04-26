from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from sklearn.metrics import r2_score
import numpy as np
from monty.serialization import loadfn
from smol.cofe import StructureWrangler
from monty.serialization import loadfn, dumpfn
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion, RegressionData
import os 
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from monty.serialization import loadfn
from ase.io import read
from smol.cofe import ClusterSubspace
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import copy
import warnings
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from smol.io import load_work
import wandb

"""
    Scripts that demonstrate CLUSTER EXPANSION using Lasso regression
"""

cwd = os.getcwd()
directory = os.path.join(cwd, "CE-smol")


# 1. STRUCTURE: let's just load the dataset saved to file: c
cwd = os.getcwd()
file_path = os.path.join(cwd, 'CE-smol/Regularization/fitted-ce/lasso766.mson')
work = load_work(file_path)
PROPERTY = "energy"
for name, obj in work.items():
    print(f'{name}: {type(obj)}\n')

wrangler = work["StructureWrangler"]
cutoffs = {2: 7, 3: 6, 4: 6}   # Here we fix the complexity of the model 
species = [{'Mn': 0.60, 'Ni': 0.4},{'As':1.0}]
subspace = wrangler.cluster_subspace
print ('Our feature matrix has the following dimensions:',
       wrangler.feature_matrix.shape)

# 4.1 Fit a Cluster Expansion:
TRIALS = 10
TEST_SIZE = 0.20
PROPERTY = 'energy'

# set up Weights And Biases
# (silence wandb terminal outputs)
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
os.environ["WANDB_SILENT"] = "true"


wandb.login()
wandb.init(project='cpsc440 ML for cluster expansion', entity="cpsc440-ml-cluster-expansion", config = {
    "species" : species,
    "cluster_info" : cutoffs,
    "property" : PROPERTY,
    "model" : "EarlyStopping"})
TRIALS = 10
TEST_SIZE = 0.20
PROPERTY = 'energy'
tolerance = 0.001  # Tolerance for stopping
n_iter_no_change = 5  # Number of iterations with no improvement to stop

wandb.init(project="my_regression_project")
X = wrangler.feature_matrix
y = wrangler.get_property_vector(key=PROPERTY)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error as mse, r2_score
import matplotlib.pyplot as plt
import warnings

# Assuming you have defined TRIALS, TEST_SIZE, n_iter_no_change, tolerance, etc.
# Initialize a structure to store the metrics for each trial
model_metrics = {}
best_mean_rmse = float('inf')
best_pipeline = None

for trial in range(TRIALS):
    pipeline = make_pipeline(StandardScaler(), SGDRegressor(
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=n_iter_no_change,
        tol=tolerance,
        fit_intercept=True))

    neg_mse_cv_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=5)
    rmse_cv_scores = np.sqrt(-neg_mse_cv_scores)
    mean_rmse = np.mean(rmse_cv_scores)

    print(f'Trial {trial}, Cross-validated RMSE: {mean_rmse}')
    wandb.log({f"Trial {trial} RMSE": mean_rmse})

    if mean_rmse < best_mean_rmse:
        best_mean_rmse = mean_rmse
        best_pipeline = pipeline

# After identifying the best pipeline, refit it to the entire training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_pipeline.fit(X_train, y_train)

y_train_pred = best_pipeline.predict(X_train)
y_test_pred = best_pipeline.predict(X_test)

rmse_test = np.sqrt(mse(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)
r2_train  = r2_score(y_test, y_test_pred)
# Plot the predictions versus the true values for the best model
plt.figure(figsize=(10, 8)) 
plt.scatter(y_test, y_test_pred, label='Predictions', s=100)  # Increase scatter point size with `s=100`)
plt.xlabel('DFT Energy (eV)', fontsize=24)
plt.ylabel('CE Predicted Energy (eV)', fontsize=24)
plt.plot(y_test, y_test, 'k--', label="Line of perfect agreement", color="red") # Line of perfect agreement
plt.title(f'SGDRegressor',  fontsize=25)
plt.text(0.05, 0.9, f'Train $R^2"$: {r2_train:.3f}', transform=plt.gca().transAxes, fontsize=20)
plt.text(0.05, 0.85, f'Test $R^2"$: {r2_test:.3f}', transform=plt.gca().transAxes, fontsize=20)
plt.legend(loc='lower right')
plt.savefig(".././CPSC440-Project/figs/SGDRegressor766.png")


from smol.cofe import RegressionData, ClusterExpansion
from random import choice

best_pipeline.fit(X[:, 1:], y)  
# Extract the model from the pipeline
model = best_pipeline.named_steps['sgdregressor']

wvec = model.coef_

print(model.intercept_)
if model.fit_intercept:
   print("Hello")
   wvec = np.concatenate((model.intercept_, model.coef_))

y_train_pred = model.predict(X_train[:, 1:])
y_test_pred = model.predict(X_test[:, 1:])
print(f'Out-of-sample RMSE is: {np.sqrt(mse(y_test, y_test_pred))} eV/prim')
print(f'In-sample RMSE is: {np.sqrt(mse(y_train, y_train_pred ))} eV/prim')
print(f'Number of Features > 1E-5: {sum(np.abs(wvec) > 1E-5)}/{len(wvec)}')


# Now you can use wvec to create a ClusterExpansion object
reg_data = RegressionData.from_sklearn(model, X, y)
expansion = ClusterExpansion(subspace, coefficients=wvec, regression_data=reg_data)
print(expansion)
from smol.io import save_work

file_path = os.path.join(cwd, 'CE-smol/Regularization/fitted-ce/SGDRegressor766.mson')
save_work(file_path, wrangler, expansion)

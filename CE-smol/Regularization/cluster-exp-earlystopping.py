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

os.environ["WANDB_SILENT"] = "true"


# (initialize wandb project and add hyperparameters)
# NOTE: wandb API key: 966b532ea49a07fa69b9a1e34f47bc02865ea9ff
wandb.login()
wandb.init(project='cpsc440 ML for cluster expansion', entity="cpsc440-ml-cluster-expansion", config = {
    "species" : species,
    "cluster_info" : cutoffs,
    "property" : PROPERTY,
    "model" : "EarlyStopping"})
TRIALS = 10
TEST_SIZE = 0.25
PROPERTY = 'energy'
tolerance = 0.001  # Tolerance for stopping
n_iter_no_change = 5  # Number of iterations with no improvement to stop


# Start a Weights & Biases run
wandb.init(project="my_regression_project")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for trial in range(TRIALS):
        X_train, X_test, y_train, y_test = train_test_split(
            wrangler.feature_matrix, wrangler.get_property_vector(key=PROPERTY),
            test_size=TEST_SIZE
        )
        model = SGDRegressor(early_stopping=True, validation_fraction=0.1, n_iter_no_change=n_iter_no_change, tol=tolerance, fit_intercept=True)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        rmse = np.sqrt(mse(y_test, y_predict))
        print(rmse)
        # Log the RMSE to Weights & Biases
        wandb.log({"Trial RMSE": rmse})
        


wandb.finish()
model = SGDRegressor(early_stopping=True, validation_fraction=0.1, n_iter_no_change=n_iter_no_change, tol=tolerance, fit_intercept=True)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print(f'Out-of-sample RMSE is: {np.sqrt(mse(y_test, y_predict))} eV/prim')
print(f'In-sample RMSE is: {np.sqrt(mse(y_train, y_train_pred ))} eV/prim')

r2_scores = []
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
r2_scores.append((r2_train, r2_test))


plt.figure(figsize=(10, 8)) 
plt.scatter(y_test, y_test_pred)
plt.xlabel('DFT Energy (eV)', fontsize=20)
plt.ylabel('CE Predicted Energy (eV)', fontsize=20)
plt.plot(y_test, y_test, 'k--', label="Line of perfect agreement") # Line of perfect agreement
plt.title(f'Early stopping',  fontsize=25)
plt.text(0.05, 0.95, f'Train $R^2$: {r2_train:.3f}\nTest $R^2$: {r2_test:.3f}', 
         transform=plt.gca().transAxes, 
         fontsize=20, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.1))

plt.legend(loc='upper right')
plt.savefig(".././CPSC440-Project/figs/EarlyStopping766.png")

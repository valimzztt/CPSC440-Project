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

# sci-kit learn for step 4.1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import copy
import warnings
from sklearn.linear_model import BayesianRidge
from sklearn.exceptions import ConvergenceWarning


import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

"""
    Scripts that demonstrate a more involved CLUSTER EXPANSION using SMOL and Bayesian Linear Regression
"""

cwd = os.getcwd()
directory = os.path.join(cwd, "CE-smol")


# 1. STRUCTURE: first create the primitive lattice: use fractional occupancies to have a disordered primitive cell
species = [{'Mn': 0.60, 'Ni': 0.4},{'As':1.0}]
my_lattice = Lattice.from_parameters(3.64580405, 3.64580405, 5.04506600, 90, 90, 120)
prim = Structure.from_spacegroup(194, my_lattice,  species, coords=[[0, 0, 0],[0.33333333, 0.66666667, 0.25]], site_properties={"oxidation": [0, 0]})
supercell = prim *(8,8,8)
# 1. STRUCTURE: We read the MnNiAs supercell from a cif file to get the structure of interest 
atoms = read("MnNiAs.cif", format="cif")


# 2. CLUSTER SUBSPACE: Specify cluster subspace information: create a cluster subspace including pair, triplet and quadruplet clusters up to given cluster diameter cutoffs.
# They keys are the number of atoms in a cluster (pair, triplet, quadruplet) and the values are the maximum diameters of the given n-body cluster (in Angstrom = 10^-10 m)

cutoffs = {2: 7, 3: 6, 4: 6}   # This is equal to CLEASE'S max_cluster_diameter =(7,6,6)  
subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)

# 3. TRAINING SET: The json file contains the structures with their corresponding ground-state energies 
energies_file = os.path.join(directory,"TrainingSet\MnNiAs-initstruct-energy-all.json" )
entries = loadfn(energies_file)

# TRAINING SET: The Wrangler object will contain the training structures with their corresponding ground-state energies 
wrangler = StructureWrangler(subspace)
for entry in entries:
    wrangler.add_entry(entry)



# 4. MACHINE LEARNING STEP: All training examples are in the wrangler object
#   a) wrangler.feature_matrix contains the feature matrix of training examples: dim(wrangler.feature_matrix)=clusters we are using to describe energy 
#   b) wrangler.get_property_vector("energy") is a vector containing the target property = energy (in eV)
#   IMPORTANT: Past this point, we could explore all possible ML/fitting algorithms because all our training data will be contained 
#   in wrangler.feature_matrix and wrangler.get_property_vector("energy")

# Tells you the dimension of your training set and how many features we have (number of clusters that within the constraints we have set (cluster size / cluster diameter))
print ('Our feature matrix has the following dimensions:',
       wrangler.feature_matrix.shape)

# 4.1 Fit a Cluster Expansion:

TRIALS = 100
TEST_SIZE = 0.20
PROPERTY = 'energy'

# set up Weights And Biases

# (silence wandb terminal outputs)
import os
os.environ["WANDB_SILENT"] = "true"

# (initialize wandb project and add hyperparameters)
# NOTE: wandb API key: 966b532ea49a07fa69b9a1e34f47bc02865ea9ff
wandb.login()
wandb.init(project='cpsc440 ML for cluster expansion', entity="cpsc440-ml-cluster-expansion", config = {
    "species" : species,
    "cluster_info" : cutoffs,
    "property" : PROPERTY,
    "model" : "BLR_fresh",
    "energy file" : energies_file})

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    rmse_list = []
    for trial in range(TRIALS):
        X_train, X_test, y_train, y_test = train_test_split(
            wrangler.feature_matrix, wrangler.get_property_vector(key=PROPERTY),
            test_size=TEST_SIZE
        )
        # We are using bayesian linear regression (note: Gaussian assumption ~ L2 reg. instead of L1)
        model = BayesianRidge(fit_intercept=True)
        # remove the constant correlation since we are fitting the intercept
        model.fit(X_train[:, 1:], y_train)
        wvec = np.concatenate((np.array([model.intercept_]),
                                model.coef_),
                                axis=0)
        y_predict = np.dot(X_test, wvec)
        rmse = np.sqrt(mse(y_test, y_predict))
        rmse_list.append(rmse)
        wandb.log({"mean test rmse": rmse, "alpha": model.lambda_}) # note: model.lambda is the regularization parameter. ignore the change of names

    mean_rmse = np.mean(rmse_list)

    # log to wandb
    wandb.config.update({"lowest_rmse": mean_rmse, "best alpha": model.lambda_}, allow_val_change = True)

wandb.finish()

# Load plotting tools to examine how the fitting is going
# '''
import matplotlib as mpl
import matplotlib.pyplot as plt

wvec = np.concatenate(
    (np.array([model.intercept_]), model.coef_),
    axis=0
)

y_predict = np.dot(X_test, wvec)
y_train_predict = np.dot(X_train, wvec)
print(f'Out-of-sample RMSE is: {np.sqrt(mse(y_test, y_predict))} eV/prim')
print(f'In-sample RMSE is: {np.sqrt(mse(y_train, y_train_predict))} eV/prim')
print(f'Number of Features > 1E-5: {sum(np.abs(wvec) > 1E-5)}/{len(wvec)}')

first_pair = subspace.orbits_by_size[2][0].bit_id
print(f'Point correlation coefficients: {wvec[:first_pair]}')
# plot the coefficients (excluding those for points))
plt.stem(range(len(wvec) - first_pair), wvec[first_pair:],
         linefmt='-', markerfmt=' ')#, basefmt=' ')
plt.xlabel('Coefficient index (i in $w_i$)')
plt.ylabel('Magnitude |$w_i$| eV/prim')


# 4.2 Generate the Cluster Expansion Object: 
from smol.cofe import RegressionData, ClusterExpansion
from random import choice

reg_data = RegressionData.from_sklearn(
    model, wrangler.feature_matrix, wrangler.get_property_vector(key=PROPERTY)
)

expansion = ClusterExpansion(
    subspace, coefficients=wvec, regression_data=reg_data
)

structure = choice(wrangler.structures)
prediction = expansion.predict(structure, normalized=True)
print(f"Structure with composition {structure.composition} has predicted energy {prediction} eV/prim")


# We have built the cluster expansion: Now let’s run canonical MC on Mn0.6Ni0.4As 

# '''
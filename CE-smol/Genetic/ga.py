import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from tpot import TPOTRegressor
from monty.serialization import loadfn
from smol.cofe import StructureWrangler, ClusterSubspace
from pymatgen.core import Lattice, Structure

"""
    Scripts that demonstrate CLUSTER EXPANSION: here we are using the Genetic algorithm keeping the cluster size fixed to 
    2:7, 3: 6, 4: 6
"""

cwd = os.getcwd()
directory = os.path.join(cwd, "CE-smol")

# 1. STRUCTURE: create the primitive lattice with fractional occupancies for a disordered cell
species = [{'Mn': 0.60, 'Ni': 0.4}, {'As': 1.0}]
my_lattice = Lattice.from_parameters(3.64580405, 3.64580405, 5.04506600, 90, 90, 120)
prim = Structure.from_spacegroup(194, my_lattice, species, coords=[[0, 0, 0], [0.33333333, 0.66666667, 0.25]])

# 2. CLUSTER SUBSPACE: create a cluster subspace with given cluster diameter cutoffs
cutoffs = {2: 7, 3: 6, 4: 6}
subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)

# 3. TRAINING SET: Load training set containing structures and their energies
energies_file = os.path.join(directory, "TrainingSet/MnNiAs-initstruct-energy-all.json")
entries = loadfn(energies_file)

# Wrangle the data
wrangler = StructureWrangler(subspace)
for entry in entries:
    wrangler.add_entry(entry)
X = wrangler.feature_matrix
y = wrangler.get_property_vector("energy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define genetic algorithm parameters
ga_model = TPOTRegressor(
    generations=5,  # Increase for better performance
    population_size=50,
    offspring_size=50,
    mutation_rate=0.9,
    crossover_rate=0.1,
    scoring='neg_mean_squared_error',
    cv=5,
    verbosity=2,
    random_state=42
)

ga_model.fit(X_train, y_train)
y_pred = ga_model.predict(X_test)
rmse = np.sqrt(mse(y_test, y_pred))
print(f'Out-of-sample RMSE is: {rmse} eV/prim')

# Export the best pipeline
ga_model.export('best_pipeline.py')

import joblib
joblib.dump(ga_model.fitted_pipeline_, 'cluster_exp_ga_model.pkl')

print(ga_model.fitted_pipeline_)




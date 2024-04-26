import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from tpot import TPOTRegressor
from monty.serialization import loadfn
from smol.cofe import StructureWrangler, ClusterSubspace
from pymatgen.core import Lattice, Structure
import joblib

"""
    Scripts that demonstrate CLUSTER EXPANSION: here we are using the Genetic algorithm keeping as degrees of freedom the cluster size as well
"""

cwd = os.getcwd()
directory = os.path.join(cwd, "CE-smol")

species = [{'Mn': 0.60, 'Ni': 0.4}, {'As': 1.0}]
my_lattice = Lattice.from_parameters(3.64580405, 3.64580405, 5.04506600, 90, 90, 120)
prim = Structure.from_spacegroup(194, my_lattice, species, coords=[[0, 0, 0], [0.33333333, 0.66666667, 0.25]])

# Define a range of cutoffs to explore
cutoff_ranges = {
    2: [5, 6, 7,8],
    3: [5, 6, 7],
    4: [0,1,2,3,4, 5, 6,7]
}

# Load training data
energies_file = os.path.join(directory, "TrainingSet/MnNiAs-initstruct-energy-all.json")
entries = loadfn(energies_file)

best_rmse = float('inf')
best_diameters = None
best_model = None

# Iterate over all combinations of cluster diameters
for cutoff_2 in cutoff_ranges[2]:
    for cutoff_3 in cutoff_ranges[3]:
        for cutoff_4 in cutoff_ranges[4]:
            cutoffs = {2: cutoff_2, 3: cutoff_3, 4: cutoff_4}
            subspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)
            wrangler = StructureWrangler(subspace)
            for entry in entries:
                wrangler.add_entry(entry)
            X = wrangler.feature_matrix
            y = wrangler.get_property_vector("energy")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            ga_model = TPOTRegressor(
                generations=5,
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

            if rmse < best_rmse:
                best_rmse = rmse
                best_diameters = cutoffs
                best_model = ga_model

print(f'Best RMSE: {best_rmse}, Best diameters: {best_diameters}')
ga_model.export('best_pipeline.py')
joblib.dump(best_model.fitted_pipeline_, 'cluster_exp_ga_model.pkl')

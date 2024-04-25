from smol.io import load_work
import os 


"""
    Scripts that demonstrate how to load a fitted Cluster Expansion saved to file or to load the dataset faster
"""

cwd = os.getcwd()
file_path = os.path.join(cwd, 'CE-smol/Regularization/fitted-ce/lasso766.mson')
work = load_work(file_path)
PROPERTY = "energy"
for name, obj in work.items():
    print(f'{name}: {type(obj)}\n')

wrangler = work["StructureWrangler"]
print(wrangler.feature_matrix)
print(wrangler.get_property_vector(key=PROPERTY))

subspace = wrangler.cluster_subspace
print(subspace)
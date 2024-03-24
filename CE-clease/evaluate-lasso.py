
from clease.settings import Concentration
import os 
from clease.settings import Concentration
from clease import Evaluate

curr_directory = os.getcwd()

#define concentration range of all elements
conc = Concentration(basis_elements=[['Mn', 'Ni'], ['As']])
conc.set_conc_ranges(ranges=[[(0,1),(0,1)], [(1,1)]])

#define crystal structure
from clease.settings import CECrystal

""" This defines the settings of the cluster expansion:
International Spacegroup number is 196
Basis vector defines the cartesian coordinates of the atoms
Hexagonal cell  """
settings = CECrystal(concentration=conc,
    spacegroup=194,
    basis=[(0.00000, 0.00000, 0.00000), (0.33333333, 0.66666667, 0.25)],
    cell=[3.64580405, 3.64580405,   5.04506600, 90, 90, 120],
    supercell_factor=8,
    db_name="clease_MnNiAs.db",
    basis_func_type='binary_linear',
    max_cluster_dia=(7,6,6))


eva = Evaluate(settings=settings, scoring_scheme='loocv', nsplits=10)
# scan different values of alpha and return the value of alpha that yields

import clease.plot_post_process as pp
import matplotlib.pyplot as plt
import json

""" Scan different values of alpha and return the value of alpha that yields
the lowest CV score """
eva.set_fitting_scheme(fitting_scheme='lasso')
alpha = eva.plot_CV(alpha_min=1E-6, alpha_max=1.0, num_alpha=50)
print("the chosen value of alpha is")
print(alpha)

# set the alpha value with the one found above, and fit data using lasso regression
eva.set_fitting_scheme(fitting_scheme='lasso', alpha=alpha)
eva.plot_fit(interactive=False)

# plot ECI values
eva.plot_ECI()

# save a dictionary containing cluster names and their ECIs
eva.save_eci(fname='eci_lasso-new.json')

fig = pp.plot_fit(eva)
alpha_image = os.path.join(curr_directory,'alpha-fit-lasso.png')
fig.savefig(alpha_image)

# plot ECI values
fig = pp.plot_eci(eva)
eci_image = os.path.join(curr_directory,'ECI-values-lasso.png')
fig.savefig(eci_image)



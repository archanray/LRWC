import sys
sys.path.append('..')
import portpy.photon as pp
import matplotlib.pyplot as plt
import numpy as np

# specify the patient data location.
data_dir = r'../data'
# Use PortPy DataExplorer class to explore PortPy data
data = pp.DataExplorer(data_dir=data_dir)
# Pick a patient 
data.patient_id = 'Lung_Patient_6'
# Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
ct = pp.CT(data)
structs = pp.Structures(data)
beams = pp.Beams(data)
# Pick a protocol
protocol_name = 'Lung_2Gy_30Fx'
# Load clinical criteria for a specified protocol
clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

# Load hyper-parameter values for optimization problem for a specified protocol
opt_params = data.load_config_opt_params(protocol_name=protocol_name)
# Create optimization structures (i.e., Rinds) 
structs.create_opt_structures(opt_params=opt_params, clinical_criteria=clinical_criteria)
# Load influence matrix
inf_matrix_sparse = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

# Create a plan using ct, structures, beams and influence matrix, and clinical criteria
plan_sparse = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix_sparse, clinical_criteria=clinical_criteria)

# Create cvxpy problem using the clinical criteria and optimization parameters
opt = pp.Optimization(plan_sparse, opt_params=opt_params, clinical_criteria=clinical_criteria)
opt.create_cvxpy_problem()
# Solve the cvxpy problem using Mosek
sol_sparse = opt.solve(solver='MOSEK', verbose=False)

# Calculate the dose using the sparse matrix
dose_sparse_1d = plan_sparse.inf_matrix.A @ (sol_sparse['optimal_intensity'] * plan_sparse.get_num_of_fractions())

# create plan_full object by specifying load_inf_matrix_full=True
beams_full = pp.Beams(data, load_inf_matrix_full=True)
# load influence matrix based upon beams and structure set
inf_matrix_full = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_full, is_full=True)
plan_full = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix = inf_matrix_full, clinical_criteria=clinical_criteria)
# use the full influence matrix to calculate the dose for the plan obtained by sparse matrix
dose_full_1d = plan_full.inf_matrix.A @ (sol_sparse['optimal_intensity'] * plan_full.get_num_of_fractions())

# Visualize the DVH discrepancy
struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
fig, ax = plt.subplots(figsize=(12, 8))
# Turn on norm flag for same normalization for sparse and full dose.
ax = pp.Visualization.plot_dvh(plan_sparse, dose_1d=dose_sparse_1d, struct_names=struct_names, style='solid', ax=ax, norm_flag=True)
ax = pp.Visualization.plot_dvh(plan_full, dose_1d=dose_full_1d, struct_names=struct_names, style='dotted', ax=ax, norm_flag=True)
ax.set_title('- Sparse    .. Full')
plt.show()
print('Done')
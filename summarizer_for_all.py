import os, pickle, sys
from src.visualization import Visualization
import matplotlib.pyplot as plt
import argparse
import configparser
import portpy.photon as pp
import numpy as np
from bs4 import UnicodeDammit

Config = configparser.ConfigParser()
Config.read("config.ini")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--method', type=str, choices=['Naive', 'AHK06', 'AKL13', 'DZ11', 'RMR', 'modifiedBKKS21', 'modifiedBKKS21-123', 'heavyRMR', 'noSparse'], help='The name of method.', default='modifiedBKKS21'
)
parser.add_argument(
    '--patient', type=str, help='Patient\'s name', default='Prostate_Patient_2'
)
parser.add_argument(
    '--threshold', type=float, help='The threshold using for the input of algorithm.', default=0.05
)
parser.add_argument(
    '--solver', type=str, default='MOSEK', help='The name of solver for solving the optimization problem'
)

parser.add_argument(
    '--samples', type=int, default=13858053, help="number of samples to grab"
)

args = parser.parse_args()
path = "outputs/medical_"+args.patient
files = []
header = str(args.patient)+"_"+str(args.method)+"_"+str(args.threshold)+"_"+str(args.samples)+"_"
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and header in i:
        files.append(os.path.join(path,i))

dose_1ds = []
dose_fulls = []
for filename in files:
    file_handler = open(filename, "rb")
    dose_1d, dose_full = pickle.load(file_handler)
    file_handler.close()
    dose_1ds.append(dose_1d)
    dose_fulls.append(dose_full)
    
    
root_folder = Config.get('Database', 'Network_Folder')
data = pp.DataExplorer(data_dir=root_folder+"/data/")
# Pick a patient
data.patient_id = args.patient
# Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
ct = pp.CT(data)
structs = pp.Structures(data)
beams = pp.Beams(data)
# Pick a protocol
pats_prot = {'Lung_Patient': 'Lung_2Gy_30Fx', 'Paraspinal_Patient': 'Paraspinal_1Fx', 'Prostate_Patient': 'Prostate_5Gy_5Fx'}
for key in pats_prot.keys():
    if key in args.patient:
        patient_key = key
        break
protocol_name = pats_prot[patient_key] #'Lung_2Gy_30Fx' #
# Load clinical criteria for a specified protocol
clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)
# Load hyper-parameter values for optimization problem for a specified protocol
opt_params = data.load_config_opt_params(protocol_name=protocol_name)
# Create optimization structures (i.e., Rinds)
structs.create_opt_structures(opt_params=opt_params, clinical_criteria=clinical_criteria)
# create plan_full object by specifying load_inf_matrix_full=True
beams_full = pp.Beams(data, load_inf_matrix_full=True)
# load influence matrix based upon beams and structure set
inf_matrix_full = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_full, is_full=True)
plan_full = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix_full, clinical_criteria=clinical_criteria)

if "Prostate_Patient" in args.patient:
    struct_names = ['PTV', 'BLADDER', 'FEMURS', 'RECTUM', 'URETHRA']
else:
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
fig, ax = plt.subplots(figsize=(12, 8))
ax = Visualization.plot_robust_dvh(plan_full, dose_1d_list=dose_fulls, struct_names=struct_names, style='solid', ax=ax, norm_flag=True, font_size=14, plot_scenario='mean')
ax = Visualization.plot_robust_dvh(plan_full, dose_1d_list=dose_1ds , struct_names=struct_names, style='dotted', ax=ax, norm_flag=True, font_size=14, plot_scenario='mean')
plt.savefig("Figures/dvhs/"+str(args.method) + "_" + str(args.threshold) + "_" + str(args.patient) + "_" + str(args.samples) + ".pdf")

############################################################# measurements #############################################################
path = "./logs/"
files = []
header = "mBSSK21_PP1_tsp1_"
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and header in i:
        files.append(os.path.join(path,i))

def find_vals(line_header, contents):
    for line in contents:
        line = line.strip("\n")
        if line_header in line:
            val = line.split(":")[-1]
            val = val.strip(" ")
            return val
    return None

values_to_find = ["number of non-zeros of the original matrix",
                  "number of non-zeros of the sparsed matrix",
                  "relative L2 norm (%)",
                  "feasibility violation",
                  "feasibility violation for PTV",
                  "relative dose discrepancy (%)",
                  "relative dose discrepancy of PTV (%)",
                  "min true dose for PTV",
                  "min approx dose for PTV"]

values = np.zeros((len(files), len(values_to_find)))

with open(files[0], "rb") as file_:
    content = file_.read()
file_encoding_format = UnicodeDammit(content).original_encoding
print(file_encoding_format, type(file_encoding_format))

for i in range(len(files)):
    file_ = files[i]
    print(file_)
    with open(file_, "rb") as fp:
        content = fp.read()
    file_encoding_format = UnicodeDammit(content).original_encoding
    file_pointer = open(file_, "r", encoding=file_encoding_format)
    contents = file_pointer.readlines()
    for j in range(len(values_to_find)):
        obs = find_vals(values_to_find[j], contents)
        values[i,j] = float(obs)
    file_pointer.close()

# aggregate measurements
mean_vals = np.mean(values, axis=0)
std_vals = np.std(values, axis=0)

for i in range(len(values_to_find)):
    print(values_to_find[i], "--mean", mean_vals[i], "--std", std_vals[i])

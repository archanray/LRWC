import os, sys, pickle
from src.visualization import Visualization
import matplotlib.pyplot as plt
import argparse
import configparser
import portpy as pp

Config = configparser.ConfigParser()
Config.read("config.ini")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--method', type=str, choices=['Naive', 'AHK06', 'AKL13', 'DZ11', 'RMR', 'modifiedBKKS21', 'heavyRMR'], help='The name of method.', default='RMR'
)
parser.add_argument(
    '--patient', type=str, help='Patient\'s name', default='Lung_Patient_5'
)
parser.add_argument(
    '--threshold', type=float, help='The threshold using for the input of algorithm.', default=0.05
)
parser.add_argument(
    '--solver', type=str, default='MOSEK', help='The name of solver for solving the optimization problem'
)

parser.add_argument(
    '--samples', type=int, default=4000000, help="number of samples to grab"
)

args = parser.parse_args()
path = "outputs/medical_"+args.patient
files = []
header = args.patient+"_"+args.method+"_"+args.threshold+"_"+args.samples+"_"
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
    
    
root_folder = Config.get('Database', 'Folder')
data = pp.DataExplorer(data_dir=root_folder+"/data/")
# Pick a patient
data.patient_id = args.patient
# Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
ct = pp.CT(data)
structs = pp.Structures(data)
beams = pp.Beams(data)
# Pick a protocol
# pats_prot = {'Lung_Patient_16': 'Lung_2Gy_30Fx', 'Paraspinal_Patient_1': 'Paraspinal_1Fx', 'Prostate_Patient_1': 'Prostate_26Fx'}
protocol_name = 'Lung_2Gy_30Fx' # 
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

struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
fig, ax = plt.subplots(figsize=(12, 8))
ax = Visualization.plot_robust_dvh(plan_full, dose_1d=dose_1ds , struct_names=struct_names, style='solid', ax=ax, norm_flag=True, font_size=14)
ax = Visualization.plot_robust_dvh(plan_full, dose_1d=dose_fulls, struct_names=struct_names, style='dotted', ax=ax, norm_flag=True, font_size=14)
plt.savefig("Figures/dvhs/"+str(args.method) + "_" + str(args.threshold) + "_" + str(args.patient) + "_" + str(args.samples) + ".pdf")
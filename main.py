import portpy.photon as pp
import src.sparsifiers as algorithms
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import sys
import time
# import optimization
import cvxpy as cp
import configparser
import os, pickle
from src.utils import fast_spectral_norm

Config = configparser.ConfigParser()
Config.read("config.ini")


def objective_function_value(x):
    obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
    obj = 0
    for i in range(len(obj_funcs)):
        if obj_funcs[i]['type'] == 'quadratic-overdose':
            if obj_funcs[i]['structure_name'] in opt.my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                if len(inf_matrix_full.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                    continue
                key = opt.matching_keys(obj_funcs[i], 'dose')
                dose_gy = opt.dose_to_gy(key, obj_funcs[i][key]) / clinical_criteria.get_num_of_fractions()
                dO = np.maximum(A[inf_matrix_full.get_opt_voxels_idx(struct), :] @ x - dose_gy, 0)
                obj += (1 / len(inf_matrix_full.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * np.sum(dO ** 2))
        elif obj_funcs[i]['type'] == 'quadratic-underdose':
            if obj_funcs[i]['structure_name'] in opt.my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                if len(inf_matrix_full.get_opt_voxels_idx(struct)) == 0:
                    continue
                key = opt.matching_keys(obj_funcs[i], 'dose')
                dose_gy = opt.get_num(obj_funcs[i][key]) / clinical_criteria.get_num_of_fractions()
                dU = np.minimum(A[inf_matrix_full.get_opt_voxels_idx(struct), :] @ x - dose_gy, 0)
                obj += (1 / len(inf_matrix_full.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * np.sum(dU ** 2))
        elif obj_funcs[i]['type'] == 'quadratic':
            if obj_funcs[i]['structure_name'] in opt.my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                if len(inf_matrix_full.get_opt_voxels_idx(struct)) == 0:
                    continue
                obj += (1 / len(inf_matrix_full.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * np.sum((A[inf_matrix_full.get_opt_voxels_idx(struct), :] @ x) ** 2))
        elif obj_funcs[i]['type'] == 'smoothness-quadratic':
            [Qx, Qy, num_rows, num_cols] = opt.get_smoothness_matrix(inf_matrix.beamlets_dict)
            smoothness_X_weight = 0.6
            smoothness_Y_weight = 0.4
            obj += obj_funcs[i]['weight'] * (smoothness_X_weight * (1 / num_cols) * np.sum((Qx @ x) ** 2) +
                                                    smoothness_Y_weight * (1 / num_rows) * np.sum((Qy @ x) ** 2))
    print("objective function value:", obj)

def l2_norm(matrix):
    return fast_spectral_norm(matrix)

if __name__ == '__main__':
    import argparse

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
    
    print("Algorithm:", args.method)
    # Use PortPy DataExplorer class to explore PortPy data
    root_folder = Config.get('Database', 'Network_Folder')
    data = pp.DataExplorer(data_dir=root_folder+"/data/")
    # Pick a patient
    data.patient_id = args.patient
    # Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)
    # Pick a protocol
    pats_prot = {'Lung_Patient': 'Lung_2Gy_30Fx', 'Paraspinal_Patient': 'Paraspinal_1Fx', 'Prostate_Patient': 'Prostate_26Fx'}
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
    # Load influence matrix
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    opt_full = pp.Optimization(plan_full, opt_params=opt_params, clinical_criteria=clinical_criteria)
    opt_full.create_cvxpy_problem()

    A = inf_matrix_full.A
    print("number of non-zeros of the original matrix: ", len(A.nonzero()[0]))
    
    method = getattr(algorithms, args.method)
    
    if args.method != "modifiedBKKS21":
        B = method(A, args.threshold)
    else:
        B = method(A, args.samples)
    print("number of non-zeros of the sparsed matrix: ", len(B.nonzero()[0]))
    print("relative L2 norm (%): ", l2_norm(A - B) / l2_norm(A) * 100)

    inf_matrix.A = B
    plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria)
    opt = pp.Optimization(plan, opt_params=opt_params, clinical_criteria=clinical_criteria)
    opt.create_cvxpy_problem()
    if args.solver == "SCIPY":
        x = opt.solve(solver=cp.SCIPY, scipy_options={"method": "highs"}, verbose=True)
    else:
        x = opt.solve(solver=args.solver, verbose=False)

    opt_full.vars['x'].value = x['optimal_intensity']
    violation = 0
    for constraint in opt_full.constraints[2:]:
        violation += np.sum(constraint.violation())
    print("feasibility violation:", violation)
    print("feasibility violation for PTV:", np.sum(opt.constraints[3].violation()))
    objective_function_value(x['optimal_intensity'])

    dose_1d = B @ (x['optimal_intensity'] * plan.get_num_of_fractions())
    dose_full = A @ (x['optimal_intensity'] * plan.get_num_of_fractions())
    print("relative dose discrepancy (%): ", (np.linalg.norm(dose_full - dose_1d) / np.linalg.norm(dose_full)) * 100)
    
    ptv_vox = inf_matrix.get_opt_voxels_idx('PTV')
    print("relative dose discrepancy of PTV (%): ", (np.linalg.norm(dose_full[ptv_vox] - dose_1d[ptv_vox]) / np.linalg.norm(dose_full[ptv_vox])) * 100)
    
    print("min true dose for PTV:", np.min(dose_full[ptv_vox]))
    print("min approx dose for PTV:", np.min(dose_1d[ptv_vox]))
    
    # save trial_dose_tildes, trial_dose_full in pickle
    folder = "./outputs/medical_"+args.patient
    if not os.path.isdir(folder):
        os.makedirs(folder)
    timestr = time.strftime("%Y%m%d-%H%M%S")    
    savefilename = folder+"/"+str(args.patient)+"_"+str(args.method)+"_"+str(args.threshold)+"_"+str(args.samples)+"_"+str(timestr)+".pkl"
    file_handler = open(savefilename, "wb")
    pickle.dump([dose_1d, dose_full], file_handler)
    file_handler.close()
    
    # VISUALIZATION ONLY    
    # struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    
    # fig, ax = plt.subplots(figsize=(12, 8))
    # # Turn on norm flag for same normalization for sparse and full dose.
    # # ax = pp.Visualization.plot_dvh(plan, dose_1d=dose_1d , struct_names=struct_names, style='solid', ax=ax, norm_flag=True)
    # # ax = pp.Visualization.plot_dvh(plan_full, dose_1d=dose_full, struct_names=struct_names, style='dotted', ax=ax, norm_flag=True)
    # plt.savefig("Figures/dvhs/"+str(args.method) + "_" + str(args.threshold) + "_" + str(args.patient) + "_" + str(args.samples) + ".pdf")
        
    # print("total time to run code:", time.time() - time_start)

import numpy as np
import portpy.photon as pp
from tqdm import tqdm
from src.sparsifiers import modifiedBKKS21
import os
import matplotlib.pyplot as plt

class TestMethods:
    def loadData(self, patient_id):
        ############### FLUFF FOR DATA IMPORT #################################
        # specify the patient data location.
        data_dir = r'../data'
        # Use PortPy DataExplorer class to explore PortPy data
        data = pp.DataExplorer(data_dir=data_dir)
        # Pick a patient 
        data.patient_id = patient_id #'Lung_Patient_6'
        # Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
        ct = pp.CT(data)
        structs = pp.Structures(data)
        beams_full = pp.Beams(data, load_inf_matrix_full=True)
        ########################################################################
        ############################## ACTUAL DATA #############################
        inf_matrix_full = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_full, is_full=True)
        ########################################################################
        return inf_matrix_full.A
    
    def mapStrToFunc(self, name):
        if name == "modBKKS21":
            return modifiedBKKS21
        return None
    
    def matrixSparsifier(self, data_matrix, A_norm, sample_sizes, method):
        # parameters for experiments
        trials = 5
        errors_per_trial = np.zeros((trials, len(sample_sizes))).astype(float)
        
        # for now compare as sparsification vs spectral norm
        # in future look to compare:
        # 1. sparsification vs LRA
        # 2. sparsification vs stable rank of the matrix
        for t in tqdm(range(trials)):
            for s in range(len(sample_sizes)):
                # sparsify the matrix
                sparse_mat = method(data_matrix, sample_sizes[s])
                # check error and store it
                errors_per_trial[t, s] = np.log(np.linalg.norm(data_matrix - sparse_mat, ord=2) / A_norm)
                pass
            pass
        
        # fix data for plotting
        mean_errors = np.mean(errors_per_trial, axis=0)
        errors_lo = np.percentile(errors_per_trial, q=20, axis=0)
        errors_hi = np.percentile(errors_per_trial, q=80, axis=0)
        
        return mean_errors, errors_lo, errors_hi
    
    def runExps(self, patient_id, methods=["modBKKS21"]):
        sample_sizes = np.arange(1000,5000,100)
        algos = [self.mapStrToFunc(i) for i in methods]
        
        # the following line loads a very tall matrix!
        # data_matrix = self.loadData(patient_id)
        data_matrix = np.random.random((5000, 100))
        true_spectral_norm = np.linalg.norm(data_matrix, ord=2)
        print("true spectral norm computed")
        
        # set up folder for saving plots
        save_folder = "./Figures/L2_error_plots/"
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        filename = os.path.join(save_folder, patient_id+".pdf")
        
        # run exps for all methods
        for i in tqdm(range(len(algos))):
            if algos[i] != None:
                mean_error, error_lo, error_hi = self.matrixSparsifier(data_matrix, true_spectral_norm, sample_sizes, algos[i])
                plt.plot(sample_sizes, mean_error, label=methods[i])
                plt.fill_between(sample_sizes, error_lo, error_hi, alpha=0.2)
        
        plt.legend()
        plt.ylabel("log normed L2 error")
        plt.yscale("log")
        plt.xlabel("total samples")
        plt.grid()
        plt.savefig(filename, bbox_inches="tight", dpi=200)
        plt.clf()
        plt.close()
        
        return None


if __name__ == "__main__":
    algos = ["modBKKS21"]
    TestMethods().runExps(patient_id='Lung_Patient_6', methods=algos)
import numpy as np
from tqdm import tqdm
from src.get_data_matrices import loadData
from src.sparsifiers import modifiedBKKS21, RMR, AHK06, AKL13, DZ11, AHK06_true
import os, pickle
import matplotlib
import matplotlib.pyplot as plt
import functools
from src.utils import fast_spectral_norm

class TestMethods:
    def mapStrToFunc(self, name):
        if name == "prob-abs":
            return functools.partial(modifiedBKKS21,mode="1")
        if name == "prob-abs+row":
            return functools.partial(modifiedBKKS21,mode="12", row_norm_preserve=False)
        if name == "prob-abs+row+sum":
            return functools.partial(modifiedBKKS21,mode="12", row_norm_preserve=True)
        if name == "prob-abs+row+col":
            return functools.partial(modifiedBKKS21,mode="123")
        if name == "prob-row":
            return functools.partial(modifiedBKKS21,mode="2")
        if name == "prob-mixed":
            return functools.partial(modifiedBKKS21,mode="4")
        if name == "RMR":
            return functools.partial(RMR)
        if name == "AHK06":
            return functools.partial(AHK06_true)
        if name == "DZ11":
            return functools.partial(DZ11)
        if name == "AKL13":
            return functools.partial(AKL13)
        return None
    
    def matrixSparsifier(self, data_matrix, A_norm, sample_sizes, method, log_scaled=True):
        # parameters for experiments
        trials = 5
        errors_per_trial = np.zeros((trials, len(sample_sizes))).astype(float)
        samples = np.zeros_like(errors_per_trial)
        
        # for now compare as sparsification vs spectral norm
        # in future look to compare:
        # 1. sparsification vs LRA
        # 2. sparsification vs stable rank of the matrix
        for t in range(trials):
            for s in range(len(sample_sizes)):
                # sparsify the matrix
                sparse_mat = method(data=data_matrix, size=sample_sizes[s])
                # check error and store it
                # errors_per_trial[t, s] = np.log(np.linalg.norm(data_matrix - sparse_mat, ord=2) / A_norm)
                if log_scaled:
                    errors_per_trial[t, s] = np.log(fast_spectral_norm(data_matrix - sparse_mat) / A_norm)
                else:
                    errors_per_trial[t, s] = fast_spectral_norm(data_matrix - sparse_mat) / A_norm
                # print("***********",sample_sizes[s], errors_per_trial[t,s])
                samples[t,s] = np.count_nonzero(sparse_mat)
                pass
            pass
        
        # fix data for plotting
        mean_errors = np.mean(errors_per_trial, axis=0)
        errors_lo = np.percentile(errors_per_trial, q=10, axis=0)
        errors_hi = np.percentile(errors_per_trial, q=90, axis=0)
        sample_sizes = np.mean(samples, axis=0)
        
        return mean_errors, errors_lo, errors_hi, sample_sizes
    
    def matrixSparsifierTh(self, data_matrix, A_norm, thresholds, method, log_scaled=True):
        # parameters for experiments
        trials = 5
        errors_per_trial = np.zeros((trials, len(thresholds))).astype(float)
        samples = np.zeros_like(errors_per_trial)
        
        # for now compare as sparsification vs spectral norm
        # in future look to compare:
        # 1. sparsification vs LRA
        # 2. sparsification vs stable rank of the matrix
        for t in range(trials):
            for s in range(len(thresholds)):
                # sparsify the matrix
                sparse_mat = method(data_matrix, thresholds[s])
                # check error and store it
                # errors_per_trial[t, s] = np.log(np.linalg.norm(data_matrix - sparse_mat, ord=2) / A_norm)
                if log_scaled:
                    errors_per_trial[t, s] = np.log(fast_spectral_norm(data_matrix - sparse_mat) / A_norm)
                else:
                    errors_per_trial[t, s] = fast_spectral_norm(data_matrix - sparse_mat) / A_norm
                # print("***********",sample_sizes[s], errors_per_trial[t,s])
                samples[t,s] = np.count_nonzero(sparse_mat)
                pass
            pass
        
        # fix data for plotting
        mean_errors = np.mean(errors_per_trial, axis=0)
        errors_lo = np.percentile(errors_per_trial, q=10, axis=0)
        errors_hi = np.percentile(errors_per_trial, q=90, axis=0)
        sample_sizes = np.mean(samples, axis=0)
        
        return mean_errors, errors_lo, errors_hi, sample_sizes.astype(int)
    
    # def runExps(self, datatype="medical", data_name="Lung_Patient_6", methods=["modBKKS21"], load_results=[False]):
    #     matplotlib.rcParams.update({'font.size': 16})
    #     plt.rcParams['figure.max_open_warning'] = 50
        
    #     # sample_sizes = np.arange(1000,10000,1000)
    #     sample_sizes = np.array([672427, 719008, 769392, 819558, 866404, 906736, 938821, 962624, 978720, 988800, 994626, 997689, 999148, 999771, 999969])
    #     algos = [self.mapStrToFunc(i) for i in methods]
        
    #     # the following line loads a very tall matrix!
    #     data_matrix = loadData(data_type = datatype, data_name = data_name)
    #     n,d = data_matrix.shape
    #     # true_spectral_norm = np.linalg.norm(data_matrix, ord=2)
    #     true_spectral_norm = fast_spectral_norm(data_matrix)
    #     print("approximate spectral norm computed")
        
    #     # set up folder for saving plots
    #     save_folder = "./Figures/L2_error_plots/"
    #     if not os.path.isdir(save_folder):
    #         os.makedirs(save_folder)
    #     filename = os.path.join(save_folder, data_name+".pdf")
        
    #     # run exps for all methods
    #     for i in tqdm(range(len(algos))):
    #         if algos[i] != None:
                
    #             foldername = "outputs/"+datatype+"_"+data_name+"/"
    #             if not os.path.isdir(foldername):
    #                 os.makedirs(foldername)
    #             savefilename = foldername+methods[i]+".pkl"
                
    #             if os.path.isfile(savefilename) and load_results[i] == True:
    #                 file_handler = open(savefilename, "rb")
    #                 mean_error, error_lo, error_hi = pickle.load(file_handler)
    #                 file_handler.close()
    #             else:
    #                 mean_error, error_lo, error_hi = self.matrixSparsifier(data_matrix, true_spectral_norm, sample_sizes, algos[i])
    #                 ### save values for future use and instant plots
    #                 file_handler = open(savefilename, "wb")
    #                 pickle.dump([mean_error, error_lo, error_hi], file_handler)
    #                 file_handler.close()
                        
    #             plt.plot(np.log(sample_sizes/(n*d)), mean_error, label=methods[i])
    #             plt.fill_between(np.log(sample_sizes/(n*d)), error_lo, error_hi, alpha=0.2)
        
    #     plt.legend()
    #     plt.ylabel(r"$\log(\|\mathbf{A}-\tilde{\mathbf{A}}\|_2 / \|\mathbf{A}\|_2)$")
    #     plt.xlabel(r"$\log(s / nd)$")
    #     plt.grid()
    #     plt.savefig(filename, bbox_inches="tight", dpi=200)
    #     plt.clf()
    #     plt.close()
        
    #     return None
    
    
    def compare_with_baseline(self, datatype="medical", data_name="Lung_Patient_6", methods=["modBKKS21"], load_results=[False]):
        matplotlib.rcParams.update({'font.size': 14})
        plt.rcParams['figure.max_open_warning'] = 50
        mode = "" # change to _test when testing features
        
        # thresholds for RMR
        # thresholds = np.arange(0.5, 0.2, -0.02)
        thresholds = np.arange(0.8, 0.5, -0.02)
        if len(methods) == 1 and "prob-abs+row+col" in methods:
            sample_sizes = np.arange(500,1000,100).astype(int)
        else:
            sample_sizes = np.zeros_like(thresholds).astype(int)
        total_samples = np.array([])
        algos = [self.mapStrToFunc(i) for i in methods]
        
        # the following line loads a very tall matrix!
        data_matrix = loadData(data_type = datatype, data_name = data_name)
        n,d = data_matrix.shape
        # true_spectral_norm = np.linalg.norm(data_matrix, ord=2)
        true_spectral_norm = fast_spectral_norm(data_matrix)
        print("approximate spectral norm computed:", true_spectral_norm)
        
        # set up folder for saving plots
        save_folder = "./Figures"+mode+"/L2_error_plots/"
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
            
        adder = "_".join(methods)
        filename = os.path.join(save_folder, data_name+"_"+adder+".pdf")
                        
        # run exps for all methods
        for i in tqdm(range(len(algos))):
            if algos[i] != None:
                
                foldername = "outputs"+mode+"/"+datatype+"_"+data_name+"/"
                if not os.path.isdir(foldername):
                    os.makedirs(foldername)
                savefilename = foldername+methods[i]+".pkl"
                
                if os.path.isfile(savefilename) and load_results[i] == True:
                    file_handler = open(savefilename, "rb")
                    if methods[i] == "RMR" or methods[i] == "AHK06" or methods[i] == "DZ11":
                        mean_error, error_lo, error_hi, sample_sizes = pickle.load(file_handler)
                    else:
                        mean_error, error_lo, error_hi, sample_sizes = pickle.load(file_handler)
                    file_handler.close()
                    total_samples = np.concatenate([total_samples, sample_sizes])
                    plot_x = sample_sizes
                else:
                    if methods[i] == "RMR" or methods[i] == "AHK06" or methods[i] == "DZ11":
                        mean_error, error_lo, error_hi, sample_sizes = self.matrixSparsifierTh(data_matrix, true_spectral_norm, thresholds, algos[i], log_scaled=True)
                        ### save values for future use and instant plots
                        file_handler = open(savefilename, "wb")
                        pickle.dump([mean_error, error_lo, error_hi, sample_sizes], file_handler)
                        file_handler.close()
                        total_samples = np.concatenate([total_samples, sample_sizes])
                        plot_x = sample_sizes
                    else:
                        if len(total_samples) == 0:
                            total_samples = sample_sizes
                        else:
                            total_samples = np.sort(total_samples)
                        mean_error, error_lo, error_hi, _ = self.matrixSparsifier(data_matrix, true_spectral_norm, total_samples, algos[i], log_scaled=True)
                        ### save values for future use and instant plots
                        file_handler = open(savefilename, "wb")
                        pickle.dump([mean_error, error_lo, error_hi, total_samples], file_handler)
                        file_handler.close()
                        plot_x = total_samples
                
                plt.plot(np.log(plot_x/(n*d)), mean_error, label=methods[i])
                plt.fill_between(np.log(plot_x/(n*d)), error_lo, error_hi, alpha=0.2)
        
        plt.legend()
        plt.ylabel(r"$\log(\|\mathbf{A}-\tilde{\mathbf{A}}\|_2 / \|\mathbf{A}\|_2)$")
        plt.xlabel(r"$\log(s / nd)$")
        plt.grid()
        plt.savefig(filename, bbox_inches="tight", dpi=200)
        plt.clf()
        plt.close()
        
        return None

if __name__ == "__main__":
    algos = ["AHK06", "RMR", "prob-abs+row", "prob-abs+row+sum"] # ["AHK06", "RMR", "prob-abs", "prob-abs+row", "prob-abs+row+col", "AKL13"]#, "prob-row"]
    # loader = [True, True, True, True] # len shouldbe +1 for baseline
    loader = [False]*len(algos)
    # loader[-1] = False
    # loader[-2] = False
    # data types: "random", "medical"
    # data names: "random" = {"gaussian"}, "medical"={"Lung_Patient_5", "Lung_Patient_6"}
    TestMethods().compare_with_baseline(datatype="random", data_name="gaussian", methods=algos, load_results=loader)
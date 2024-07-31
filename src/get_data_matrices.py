import numpy as np
from scipy import stats
from scipy.stats import ortho_group, truncnorm
import portpy.photon as pp

def loadData(data_type="medical", data_name="gaussian"):
    if data_type == "medical":
        ############### FLUFF FOR DATA IMPORT #################################
        # specify the patient data location.
        data_dir = r'../data'
        # Use PortPy DataExplorer class to explore PortPy data
        data = pp.DataExplorer(data_dir=data_dir)
        # Pick a patient 
        data.patient_id = data_name #'Lung_Patient_6'
        # Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
        ct = pp.CT(data)
        structs = pp.Structures(data)
        beams_full = pp.Beams(data, load_inf_matrix_full=True)
        ########################################################################
        ############################## ACTUAL DATA #############################
        inf_matrix_full = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_full, is_full=True)
        ########################################################################
        return inf_matrix_full.A / np.max(inf_matrix_full.A)
    
    if data_type == "random":
        n = 1000
        if data_name == "gaussian":
            data = np.random.normal(size=(n,n))
            data = data-np.min(data)
            data = data / np.max(data)
            data = (data + data.T) / 2
            
        if data_name == "uniform":
            data = np.random.rand(n,n)
            data = (data + data.T) / 2
        
        return data
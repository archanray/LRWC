import os

n = 5

run_args = "--method thresholdedBKKS21 --patient Prostate_Patient_2 --samples_percent 0.2 --split 01 --split_type infinity"
# "--method modifiedBKKS21 --patient Prostate_Patient_2 --samples_percent 0.2"
log_file_header = "thBSSK21_PP2_tsp0.2_sp01_st_infty_"

for i in range(1,n+1):
    # command = "python main.py --method RMR --patient Paraspinal_Patient_1 --threshold 0.05 --solver MOSEK 2>&1 | tee logs/RMR_PS1_005_"+str(i)+".txt"
    command = "python main.py " + run_args + " --solver MOSEK 2>&1 | tee logs/"+log_file_header+str(i)+".txt"
    # command = "python main.py --method noSparse --patient Paraspinal_Patient_2 --solver MOSEK 2>&1 | tee logs/nS_PS2_"+str(i)+".txt"
    os.system(command)

os.system("python summarizer_for_all.py "+run_args+" --log_file_header "+log_file_header)

os.system("git add .")
os.system("git commit -m \"automated system commit\"")
os.system("git push")
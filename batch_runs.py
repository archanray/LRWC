import os

n = 5

run_args = "--method modifiedBKKS21 --patient Prostate_Patient_2 --samples_percent 0.01"
#"--method thresholdedBKKS21 --patient Prostate_Patient_2 --samples_percent 0.2 --split 1 --split_type ell_one"
#"--method thresholdedBKKS21 --patient Prostate_Patient_2 --samples_percent 0.2 --split 10 --split_type infinity"
# "--method modifiedBKKS21 --patient Prostate_Patient_2 --samples_percent 0.2"
log_file_header = "mBSSK21_12_PP2_tsp0.01_"
#"thBSSK21_PP2_tsp0.2_sp1_st_ell1_"
#"mBSSK21_PP2_tsp0.2_"
#"thBSSK21_PP2_tsp0.2_sp10_st_infty_"

for i in range(1,n+1):
    # command = "python main.py --method RMR --patient Paraspinal_Patient_1 --threshold 0.05 --solver MOSEK 2>&1 | tee logs/RMR_PS1_005_"+str(i)+".txt"
    command = "python main.py " + run_args + " --solver MOSEK 2>&1 | tee logs/"+log_file_header+str(i)+".txt"
    # command = "python main.py --method noSparse --patient Paraspinal_Patient_2 --solver MOSEK 2>&1 | tee logs/nS_PS2_"+str(i)+".txt"
    os.system(command)

summary_filename = "summary_"+log_file_header+".txt"
os.system("python summarizer_for_all.py "+run_args+" --log_file_header "+log_file_header+ " --run_type full 2>&1 | tee logs/"+summary_filename)

os.system("git add .")
os.system("git commit -m \"automated system commit\"")
os.system("git push")
import os

n = 5

for i in range(1,n+1):
    # command = "python main.py --method RMR --patient Paraspinal_Patient_1 --threshold 0.05 --solver MOSEK 2>&1 | tee logs/RMR_PS1_005_"+str(i)+".txt"
    command = "python main.py --method modifiedBKKS21 --patient Prostate_Patient_2 --samples_percent 1.0 --solver MOSEK 2>&1 | tee logs/mBSSK21_PP1_tsp1_"+str(i)+".txt"
    # command = "python main.py --method noSparse --patient Paraspinal_Patient_2 --solver MOSEK 2>&1 | tee logs/nS_PS2_"+str(i)+".txt"
    os.system(command)
    
os.system("git add .")
os.system("git commit -m \"automated system commit\"")
os.system("git push")
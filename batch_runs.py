import os

n = 1

for i in range(1,n+1):
    # command = "python main.py --method modifiedBKKS21 --patient Paraspinal_Patient_2 --threshold 0.05 --solver MOSEK 2>&1 | tee logs/RMR_PS2_005_"+str(i)+".txt"
    # command = "python main.py --method modifiedBKKS21 --patient Paraspinal_Patient_2 --samples 3500000 --solver MOSEK 2>&1 | tee logs/mBSSK21_PS2_3500000_"+str(i)+".txt"
    command = "python main.py --method noSparse --patient Paraspinal_Patient_2 --solver MOSEK 2>&1 | tee logs/nS_PS2_"+str(i)+".txt"
    os.system(command)
    
os.system("git add .")
os.system("git commit -m \"automated system commit\"")
os.system("git push")
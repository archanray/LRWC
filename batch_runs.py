import os

for i in range(3,5):
    command = "python main.py --method modifiedBKKS21 --patient Lung_Patient_5 --samples 12000000 --solver MOSEK 2>&1 | tee logs/mBKKS21_LP5_12000000_"+str(i)+".txt"
    os.system(command)
import os

for i in range(5,6):
    command = "python main.py --method modifiedBKKS21 --patient Lung_Patient_5 --samples 12000000 --solver MOSEK 2>&1 | tee logs/mBKKS21_LP5_12000000_"+str(i)+".txt"
    os.system(command)
    
os.system("git add .")
os.system("git commit -m \"automated system commit\"")
os.system("git push")
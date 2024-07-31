## Dependencies
* `portpy` : `pip install portpy`

## Usage example for main optimization
`python main.py --method RMR --patient Lung_Patient_1 --threshold 0.05`

## Usage for l2 error plots:
`python unit_test.py`

## Usage to store logs:
append command with ` 2>&1 | tee logs/RMR_LP5_005.txt`

## Steps to create your config file (need to do this before running the code)
1. Create a file called `config.ini`
2. If your root data folder is `"../"` then update the file using the following block of code:
```
[Database]
Folder: ../
```
Save it, and then update line 15 of `main.py` accordingly.

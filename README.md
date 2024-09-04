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

## Usage for runing with multiple initializations
1. to generate outputs with multiple initialization and also store log files run: `python batch_runs.py`
2. to visualize after step (1) is complete: `python summarizer_for_all.py`. Do note, you'd need to provide exact arguments to `summarizer_for_all.py` you used when running `batch_runs`.
Currently `batch_runs` also includes `summarizer_for_all`, so you'd need step 2 only if you are doig some ablation studies.


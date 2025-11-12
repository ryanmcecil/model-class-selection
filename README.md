# Model Class Selection

Code for Model Class Selection preprint.

## Instructions to Reproduce

First, we need to ensure that the `.venv` package installs in the local directory. If not installed, install `poetry` on your system, then call
```
poetry config virtualenvs.in-project true --local
```
Next install the virtual environment
```
poetry install
```
Once the virtual environment is installed, to produce all the simulations and plots from the paper call
```
bash build_plots.sh
```
Please see the `build_plots.sh` to find which python files correspond to each figure from the paper. To control where the simulation results are saved, see the `.env` file. Also, `rl_data_tests.py` may produce an error when run because it relies on outside datasets not included in this repository. Please contact me if you want a zip file of the downloaded datasets. The code for processing the datasets can be found in `src/process_data.R`.
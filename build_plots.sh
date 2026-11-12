# Bash file to reproduce simulations and plots from paper
cd .venv/bin
source activate
cd ../..

# Figure 2
python3 src/pairwise_mars_sims.py

# Figure 3
python3 src/feature_selection_linear_rf_sims.py

# Figure 4
python3 src/additive_noise_rl_datasets.py

# Table 1 and Figure 5
python3 src/rl_data_tests.py

# Figure B1
python3 src/sui_convergence_example.py

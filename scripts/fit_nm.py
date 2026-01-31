#%% --------- imports  ---------
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pcntoolkit as ptk
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from data_utils import load_data

sns.set(style='whitegrid')

#%% --------- load data --------------
root = Path(__file__).resolve().parents[2]
print('top level dir:', root)

# modaltity: 'ct', 'sc', 'sa', 'fa'
# variant for 'ct': None, 'DK'
df, response_variables, out_dir = load_data(modality='fa', 
                                            variant=None, 
                                            top_level_dir = root)
os.makedirs(os.path.join(out_dir), exist_ok=True)

#%%  --------- configure norm data and model ---------

# set responses and covariates
covariates = ["age"]
batch_effects = ["site", "sex"]

# Remove variables with no variance
response_variables = list(
    filter(lambda x: df[x].var() > 0, response_variables)
)  
print(response_variables)

# configure norm data object
reference_norm_data = ptk.NormData.from_dataframe(
    name="pu25_thickness",
    dataframe=df,
    covariates=covariates,
    batch_effects=batch_effects,
    response_vars=response_variables,
    subject_ids='sub_id',
    remove_Nan=True,
    remove_outliers=True,
    z_threshold=10  
)

# configure blr
template_blr = ptk.BLR(
    name="template",
    basis_function_mean = ptk.BsplineBasisFunction(),
    fixed_effect = True,  
    fixed_effect_var = False,
    warp_name="WarpSinhArcsinh",
    warp_reparam = True, 
    heteroskedastic= False,  
    optimizer="powell",
    ard=False,
)

# configure normative model
model = ptk.NormativeModel(
    template_regression_model=template_blr,
    savemodel=True,
    evaluate_model=True,
    saveresults=True,
    saveplots=True,
    inscaler="standardize",
    outscaler="standardize",
    save_dir= out_dir,
)

# configure runner
venv_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable)))
runner = ptk.Runner(
    cross_validate=False,
    parallelize=True, 
    n_batches = len(reference_norm_data.response_vars), 

    environment=venv_path,
    job_type="slurm", 
    time_limit="12:00:00",
    memory = "2GB",
    n_cores=1,
    log_dir=os.path.join(out_dir,'logs/'),
    temp_dir=os.path.join(out_dir,'tmp/'),
    #preamble = "module load gcc/13.3.0; module load anaconda3" 
)

#%% ---------- configure train/test split ---------

# configure train test split
train, test = reference_norm_data.train_test_split(
    splits=(0.5, 0.5), split_names=["train", "test"], random_state=42
)
print('train:', len(train.observations))
print('test:', len(test.observations))

#%% ---------- fit and predict (single thread ---------

model.fit_predict(train, test)

#%% ---------- fit model (single thread) -------

model.fit(reference_norm_data)

#%% ---------- fit model (multiple threads) -------

runner.fit(model, reference_norm_data, observe=False)

#%% ---------- fit predict model (multiple threads) -------

runner.fit_predict(model, train, train, observe=False)
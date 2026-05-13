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

modality = 'ct' # 'ct', 'sc', 'sa', 'fa', 'fc'
variant = None # for ct and sa : None, 'DK' - otherwise unused
model_type = 'BLRw' #BLRw or HBR_Sb

df, response_variables, out_dir = load_data(modality=modality, 
                                            variant=variant,
                                            model_type=model_type,
                                            top_level_dir = root)
os.makedirs(os.path.join(out_dir), exist_ok=True)

if modality == 'sa'and variant==None: #if Destrieux SA
    df = df[(df == 0).sum(axis=1) < 25] #temporary - drop ukb subjects with freesurfer processing issue for Destrieux

# if modality == 'sa' :
#     name = 'surfacearea'
#     splits=(0.5, 0.5)
# elif modality == 'fc': #less subjects so change split
#     name = 'connectomes'
#     splits=(0.8, 0.2)
if modality == 'fc':
    splits=(0.8, 0.2)
else:
    splits=(0.5, 0.5)
#%%  --------- configure norm data ---------

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
    name="pu25_"+ modality,
    dataframe=df,
    covariates=covariates,
    batch_effects=batch_effects,
    response_vars=response_variables,
    subject_ids='sub_id',
    remove_Nan=True,
    remove_outliers=True,
    z_threshold=10  
)
#%%  --------- configure model ---------
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

# configure HBR - parameters taken from JMM Bayer's HBR SHASH notebook
mu = ptk.make_prior(
    linear=True,
    slope=ptk.make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
    intercept=ptk.make_prior(
        random=True,
        mu=ptk.make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
        sigma=ptk.make_prior(dist_name="Normal", dist_params=(0.0, 1.0), mapping="softplus", mapping_params=(0.0, 3.0)),
    ),
    basis_function=ptk.BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
)
sigma = ptk.make_prior(
    linear=True,
    slope=ptk.make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
    intercept=ptk.make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
    basis_function=ptk.BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    mapping="softplus",
    mapping_params=(0.0, 3.0),
)

epsilon = ptk.make_prior(
    dist_name="Normal",
    dist_params=(0.0, 1.0),
)

delta = ptk.make_prior(
    dist_name="Normal",
    dist_params=(1.0, 1.0),
    mapping="softplus",
    mapping_params=(0.0,3.0,0.6),
)

template_hbr = ptk.HBR(
    name="template",
    cores=16,
    progressbar=True,
    draws=1500,
    tune=500,
    chains=4,
    nuts_sampler="nutpie",
    likelihood=ptk.SHASHbLikelihood(mu, sigma, epsilon, delta),
)

# configure normative model
if model_type == 'BLRw':
    template_model = template_blr
elif model_type == 'HBR_Sb':
    template_model = template_hbr
    
model = ptk.NormativeModel(
    template_regression_model=template_model,
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
    time_limit="12:00:00", #need 24h for HBR
    memory = "2GB", #need 50 to 55GB for full reference_normdata HBR Sb
    n_cores=1,
    log_dir=os.path.join(out_dir,'logs/'),
    temp_dir=os.path.join(out_dir,'tmp/'),
    #preamble = "module load gcc/13.3.0; module load anaconda3" 
)

#%% ---------- configure train/test split ---------

# configure train test split
train, test = reference_norm_data.train_test_split(
    splits=splits, split_names=["train", "test"], random_state=42
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

runner.fit_predict(model, train, test, observe=False)

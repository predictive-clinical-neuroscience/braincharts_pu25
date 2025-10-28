#%% --------- imports  ---------
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pcntoolkit as ptk
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid')

#%% --------- load data --------------

# load data
data_dir = '/project/3022054.01/projects/braincharts/data'
root_dir = '/project/3022054.01/projects/braincharts/braincharts_pu25'

# # main results - elife 2022
# df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_tr.csv'), index_col=0) 
# df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_te.csv'), index_col=0)
# #df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_patients_te.csv'), index_col=0)
# out_dir = os.path.join(root_dir,'models','lifespan_59K_82sites')

# pu25 results
df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_extended_tr.csv'), index_col=0) 
df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_extended_te.csv'), index_col=0)
# # cortical thickness
# with open(os.path.join(root_dir,'docs','phenotypes_ct_lh.txt')) as f:
#     idp_ids_lh = f.read().splitlines()
# with open(os.path.join(root_dir,'docs','phenotypes_ct_rh.txt')) as f:
#     idp_ids_rh = f.read().splitlines()
# response_variables = idp_ids_lh + idp_ids_rh
# out_dir = os.path.join(root_dir,'models','lifespan_ct_67K_89sites')
# subcortical volumes
with open(os.path.join(root_dir,'docs','phenotypes_sc.txt')) as f:
    idp_ids_sc = f.read().splitlines()
exclude = ['Left-vessel', 'Left-choroid-plexus',
           'Right-vessel', 'Right-choroid-plexus',
           'TotalGrayVol', 'SupraTentorialVolNotVent',
           'EstimatedTotalIntraCranialVol']
response_variables = idp_ids_sc
response_variables = [
    col for col in response_variables if col not in exclude
    ]
out_dir = os.path.join(root_dir,'models','lifespan_sc_67K_89sites')

# surface area
#df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_surfacearea_resample_0_tr.csv'), index_col=0) 
#df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_surfacearea_resample_0_te.csv'), index_col=0)
#df_tr.dropna(inplace=True,how='any')
#df_te.dropna(inplace=True,how='any')
#out_dir = os.path.join(root_dir,'models','surfacearea_20K_66sites_train_compact')

# Desikan-Killiany atlas
#df_tr = pd.read_csv(os.path.join(data_dir,'DK_lifespan_big_ct_tr.csv'), index_col=0) 
#df_te = pd.read_csv(os.path.join(data_dir,'DK_lifespan_big_ct_te.csv'), index_col=0)
#out_dir = os.path.join(root_dir,'models','lifespan_DK_46K_59sites_compact')

# diffusion (FA)
# df_tr = pd.read_pickle(os.path.join(data_dir,'FA_clean_train.pkl')) 
# df_te = pd.read_pickle(os.path.join(data_dir,'FA_clean_test.pkl'))
# out_dir = os.path.join(root_dir,'models','lifespan_FA_24K_19sites_train_compact')
# # fix some coding errors
# df_tr.columns = df_tr.columns.str.replace(" - ", "-")
# df_tr.columns = df_tr.columns.str.replace(" ", "_")
# df_te.columns = df_te.columns.str.replace(" - ", "-")
# df_te.columns = df_te.columns.str.replace(" ", "_")
# df_te = df_te.loc[df_te['site'] != 'DHCP_Evelina']
# df_tr = df_tr.loc[df_tr['site'] != 'DHCP_Evelina']
# df_tr['age'] = pd.to_numeric(df_tr['age'])
# df_te['age'] = pd.to_numeric(df_te['age'])

os.makedirs(os.path.join(out_dir), exist_ok=True)

df = pd.concat((df_tr, df_te))
df['sex'] = df.apply(lambda x: {0: "F", 1: "M"}[x['sex']], axis=1)
df['sub_id'] = df.index.astype(str)


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

runner.fit_predict(model, train, test, observe=False)
#%%  --------- plots  ---------

# plot data
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(y="site", data=df, ax=ax[0], hue="sex", palette="Set2", legend=False)
sns.scatterplot(
    x="age",
    y="lh_G&S_paracentral_thickness",
    data=df,
    ax=ax[1],
    hue="sex",
    palette="Set2",
)
ax[0].set_title("Site and sex distribution")
ax[1].set_title("Age and paracentral thickness")
plt.show()
df.shape

#%%  --------- nicer centile plot  ---------
# plot centiles
plotdir = os.path.join(out_dir, "plots")
ptk.util.plotter.plot_centiles(
    model,
    #covariate="age",
    #covariate_range=[0, 90],
    batch_effects='all',
    hue_data= ('batch_effects', 'site'),
    scatter_data=test,
    #show_other_data=True,
    #harmonize_data=True,
    #style="site",
    #style_order=site_ids,
    #legend_out=True
    )


#%% --------- imports  ---------
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pcntoolkit as ptk

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
with open(os.path.join(root_dir,'docs','phenotypes_ct_lh.txt')) as f:
    idp_ids_lh = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_ct_rh.txt')) as f:
    idp_ids_rh = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_sc.txt')) as f:
    idp_ids_sc = f.read().splitlines()
response_variables = idp_ids_lh + idp_ids_rh #+ idp_ids_sc
response_variables = response_variables[:5]
out_dir = os.path.join(root_dir,'models','lifespan_ct_67K_89sites')

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

#%%  --------- plotting  ---------

#generate plot 
import seaborn as sns
import matplotlib.pyplot as plt

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

#%%  --------- configure norm data  ---------
# set responses and covariates
covariates = ["age"]
batch_effects = ["site", "sex"]

# save site ids
site_ids =  sorted(set(df_tr['site'].to_list()))
with open(os.path.join(out_dir,'site_ids.txt'),'w') as f:
    f.write('\n'.join(site_ids))

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
    z_threshold=10,  # The default here is 3, but we use 10 for demonstration purposes
)

#%% --------- configure model  ---------
# configure train test split
train, test = reference_norm_data.train_test_split(
    splits=(0.5, 0.5), split_names=["train", "test"], random_state=42
)
print('train:', len(train.observations))
print('test:', len(test.observations))

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

# cofnigure normative model
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

#%% ---------- fit model  ---------

model.fit_predict(train, test)

#%% rest

site_ids =  sorted(set(df_tr['site'].to_list()))
with open(os.path.join(out_dir,'site_ids.txt'),'w') as f:
    f.write('\n'.join(site_ids))

# load the idps to process
with open(os.path.join(root_dir,'docs','phenotypes_ct_lh.txt')) as f:
    idp_ids_lh = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_ct_rh.txt')) as f:
    idp_ids_rh = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_sc.txt')) as f:
    idp_ids_sc = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_sa_lh.txt')) as f:
    idp_ids_sa_lh = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_sa_rh.txt')) as f:
    idp_ids_sa_rh = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_ct_dk_lh.txt')) as f:
    idp_ids_dk_lh = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_ct_dk_rh.txt')) as f:
    idp_ids_dk_rh = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_ct_dk_mean.txt')) as f:
    idp_ids_dk_mean = f.read().splitlines()
with open(os.path.join(root_dir,'docs','phenotypes_fa.txt')) as f:
    idp_ids_fa = f.read().splitlines()

#idp_ids = idp_ids_lh + idp_ids_rh + idp_ids_sc
#idp_ids = idp_ids_dk_lh + idp_ids_dk_rh + idp_ids_dk_mean
#idp_ids = idp_ids_sa_lh + idp_ids_sa_rh
#idp_ids = idp_ids_sc
idp_ids = idp_ids_fa
#idp_ids = ['SubCortGrayVol']
with open(os.path.join(out_dir,'idp_ids.txt'),'w') as f:
    f.write('\n'.join(idp_ids))

outlier_thresh = 7

warp =  'WarpSinArcsinh'   # 'WarpBoxCox', 'WarpSinArcsinh'  or None

# limits for cubic B-spline basis 
xmin = -5 # boundaries for ages of UKB participants +/- 5
xmax = 110

################################### RUN #######################################

for nummer, idp in enumerate(idp_ids): 
    print('scanning',nummer,' IDP:', idp)

    # configure and save the responses
    y_tr = df_tr[idp].to_numpy() 
    y_te = df_te[idp].to_numpy()
    
    # remove gross outliers
    yz_tr = (y_tr - np.mean(y_tr)) / np.std(y_tr)
    yz_te = (y_te - np.mean(y_te)) / np.std(y_te)
    nz_tr_i = np.bitwise_and(np.abs(yz_tr) < outlier_thresh, y_tr > 0)
    nz_te_i = np.bitwise_and(np.abs(yz_te) < outlier_thresh, y_te > 0) 
    if nummer == 0: 
        nz_tr = nz_tr_i
        nz_te = nz_te_i
    else:
        nz_tr = np.bitwise_and(nz_tr, nz_tr_i)
        nz_te = np.bitwise_and(nz_te, nz_te_i)
  
idp_dir = out_dir
os.chdir(idp_dir)

# configure and save response variables
y_tr = df_tr[idp_ids].to_numpy()
y_tr = y_tr[nz_tr,:]
y_te = df_te[idp_ids].to_numpy()
y_te = y_te[nz_te,:]

resp_file_tr = os.path.join(idp_dir, 'resp_tr.pkl')
resp_file_te = os.path.join(idp_dir, 'resp_te.pkl') 
ptksave(y_tr, resp_file_tr)
ptksave(y_te, resp_file_te)
    
# configure and save the covariates
X_tr = create_design_matrix(df_tr[cols_cov].loc[nz_tr], 
                            site_ids = df_tr['site'].loc[nz_tr],
                            basis = 'bspline',
                            xmin = xmin, 
                            xmax = xmax)
X_te = create_design_matrix(df_te[cols_cov].loc[nz_te], 
                            site_ids = df_te['site'].loc[nz_te],
                            all_sites=site_ids,
                            basis = 'bspline', 
                            xmin = xmin, 
                            xmax = xmax)

cov_file_tr = os.path.join(idp_dir, 'cov_bspline_tr.pkl')
cov_file_te = os.path.join(idp_dir, 'cov_bspline_te.pkl')
ptksave(X_tr, cov_file_tr)
ptksave(X_te, cov_file_te)

fit(cov_file_tr, 
    resp_file_tr, 
    alg='blr', 
    optimizer = 'l-bfgs-b', 
    savemodel='True',
    warp=warp, 
    warp_reparam=True) 

# # Make prdictsion with test data
# yhat_te, s2_te, Z = predict(cov_file_te, 
#                             alg='blr', 
#                             respfile=resp_file_te, 
#                             model_path=os.path.join(idp_dir,'Models'), 
#                             inputsuffix='_fit',
#                             outputsuffix='_predict')

    




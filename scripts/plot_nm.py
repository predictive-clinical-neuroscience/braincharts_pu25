#%%  --------- Imports  ---------
import os
import pandas as pd
import pcntoolkit as ptk

data_dir = '/project/3022054.01/projects/braincharts/data'
root_dir = '/project/3022054.01/projects/braincharts/braincharts_pu25'

#%%  --------- load data  ---------

#test = ptk.NormData.load(os.path.join(out_dir,'test_data_normdata.pkl'))

out_dir = os.path.join(root_dir,'models','lifespan_sc_67K_89sites')
df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_extended_tr.csv'), index_col=0) 
df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_extended_te.csv'), index_col=0)

df = pd.concat((df_tr, df_te))

df['sex'] = df.apply(lambda x: {0: "F", 1: "M"}[x['sex']], axis=1)
df['sub_id'] = df.index.astype(str)

# load model 
model = ptk.NormativeModel.load(out_dir)

# configure 
covariates = model.covariates
batch_effects = list(model.unique_batch_effects.keys()) 
response_variables = model.response_vars

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

#%%  --------- nicer centile plot  ---------

# plot centiles
plotdir = os.path.join(out_dir, "plots")
ptk.util.plotter.plot_centiles(
    model,
    #covariate="age",
    #covariate_range=[0, 90],
    #batch_effects='all',
    #hue_data= ('batch_effects', 'site'),
    scatter_data=reference_norm_data,
    #show_other_data=True,
    #harmonize_data=True,
    #style="site",
    #style_order=site_ids,
    #legend_out=True
    )


# %%


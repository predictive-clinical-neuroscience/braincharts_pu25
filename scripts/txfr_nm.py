import os
import pcntoolkit as ptk
import pandas as pd
import numpy as np

# load data
data_dir = '/project/3022054.01/projects/braincharts/data/'
root_dir = '/project/3022054.01/projects/braincharts/braincharts_pu25'
model_dir = os.path.join(root_dir,'models','lifespan_sc_67K_89sites')
out_dir = model_dir + '_txfr'
os.makedirs(out_dir,exist_ok=True)

model = ptk.NormativeModel.load(model_dir)
batch_effects = ["site"]
covariates = ["age"]

data_ad_path = os.path.join(data_dir, 'pcnportal_test_data','OpenNeuroTransfer_ct_ad.csv')
df_ad = pd.read_csv(data_ad_path, index_col=0)
data_te_path = os.path.join(data_dir, 'pcnportal_test_data','OpenNeuroTransfer_ct_te.csv')
df_te = pd.read_csv(data_ad_path, index_col=0)
df = pd.concat([df_ad, df_te], axis=0)
df['sub_id'] = df.index.astype(str)

norm_data = ptk.NormData.from_dataframe(
    name="pu25_thickness_txfr", 
    dataframe=df,
    batch_effects=batch_effects,
    subject_ids='sub_id',
    response_vars = model.response_vars,
    covariates = model.covariates,
    #X = np.array(df_ad.loc[:, covariates]).squeeze()
    # optional arguments
    remove_Nan=True,
    remove_outliers=True,
    z_threshold=10  
)
norm_data.register_batch_effects() 

split = [0.5, 0.5]
train, test = norm_data.train_test_split(splits = split)
model.transfer_predict(train, test, save_dir = out_dir) 
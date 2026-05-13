
import sys
import os
import numpy as np
import pandas as pd

def load_data(modality='ct', variant=None, model_type=None, top_level_dir=None):
    """Load and return the combined training and test datasets."""

    if top_level_dir is None:
        data_dir = '/project/3022054.01/projects/braincharts/data'
        root_dir = '/project/3022`054.01/projects/braincharts/braincharts_pu25'
    else:
        data_dir = os.path.join(top_level_dir,'data')
        root_dir = os.path.join(top_level_dir,'braincharts_pu25')

    # # main results - elife 2022
    # df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_tr.csv'), index_col=0) 
    # df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_te.csv'), index_col=0)
    # #df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_patients_te.csv'), index_col=0)
    # out_dir = os.path.join(root_dir,'models','lifespan_59K_82sites')
    
    # cortical thicknes (pu25 results)
    if modality.lower() == 'ct':
        if variant and variant.upper() == 'DK':
            # Desikan-Killiany atlas
            df_tr = pd.read_csv(os.path.join(data_dir,'DK_lifespan_big_ct_tr.csv'), index_col=0) 
            df_te = pd.read_csv(os.path.join(data_dir,'DK_lifespan_big_ct_te.csv'), index_col=0)
            with open(os.path.join(root_dir,'docs','phenotypes_ct_dk_lh.txt')) as f:
                idp_ids_dk_lh = f.read().splitlines()
            with open(os.path.join(root_dir,'docs','phenotypes_ct_dk_rh.txt')) as f:
                idp_ids_dk_rh = f.read().splitlines()
            with open(os.path.join(root_dir,'docs','phenotypes_ct_dk_mean.txt')) as f:
                idp_ids_dk_mean = f.read().splitlines()
            response_variables = idp_ids_dk_lh + idp_ids_dk_rh + idp_ids_dk_mean
            out_dir = os.path.join(root_dir,'models', model_type + '_lifespan_ct_dk_46K_59sites')
        else:
            # Destrieux atlas
            df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_extended_tr.csv'), index_col=0) 
            df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_extended_te.csv'), index_col=0)
            with open(os.path.join(root_dir,'docs','phenotypes_ct_lh.txt')) as f:
                idp_ids_lh = f.read().splitlines()
            with open(os.path.join(root_dir,'docs','phenotypes_ct_rh.txt')) as f:
                idp_ids_rh = f.read().splitlines()
            response_variables = idp_ids_lh + idp_ids_rh
            out_dir = os.path.join(root_dir,'models', model_type + '_lifespan_ct_67K_89sites')

    # subcortical volumes (pu25 results)
    elif modality.lower() == 'sc':
        df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_extended_tr.csv'), index_col=0) 
        df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_controls_extended_te.csv'), index_col=0)
        # add log transformed WM-hypointensities
        df_tr['log-WM-hypointensities'] = np.log1p(df_tr['WM-hypointensities'])
        df_te['log-WM-hypointensities'] = np.log1p(df_te['WM-hypointensities'])
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
        response_variables = response_variables + ['WM-hypointensities', 'log-WM-hypointensities']
        out_dir = os.path.join(root_dir,'models', model_type + '_lifespan_sc_67K_89sites')

    elif modality.lower() == 'sa':
        # surface area
        if variant and variant.upper() == 'DK':
            # Desikan-Killiany atlas
            df_tr = pd.read_csv(os.path.join(data_dir,'DK_lifespan_big_sa_tr.csv'), index_col=0) 
            df_te = pd.read_csv(os.path.join(data_dir,'DK_lifespan_big_sa_te.csv'), index_col=0)
            with open(os.path.join(root_dir,'docs','phenotypes_sa_dk_lh.txt')) as f:
                idp_ids_sa_dk_lh = f.read().splitlines()
            with open(os.path.join(root_dir,'docs','phenotypes_sa_dk_rh.txt')) as f:
                idp_ids_sa_dk_rh = f.read().splitlines()
            response_variables = idp_ids_sa_dk_lh + idp_ids_sa_dk_rh
            out_dir = os.path.join(root_dir,'models', model_type +'_lifespan_sa_dk_46K_59sites')
        else:
            # Destrieux atlas
            df_tr = pd.read_csv(os.path.join(data_dir,'lifespan_big_surfacearea_resample_0_tr.csv'), index_col=0) 
            df_te = pd.read_csv(os.path.join(data_dir,'lifespan_big_surfacearea_resample_0_te.csv'), index_col=0)
            df_tr.dropna(inplace=True,how='any')
            df_te.dropna(inplace=True,how='any')
            with open(os.path.join(root_dir,'docs','phenotypes_sa_lh.txt')) as f:
                idp_ids_sa_lh = f.read().splitlines()
            with open(os.path.join(root_dir,'docs','phenotypes_sa_rh.txt')) as f:
                idp_ids_sa_rh = f.read().splitlines()
            response_variables = idp_ids_sa_lh + idp_ids_sa_rh
            out_dir = os.path.join(root_dir,'models', model_type +'_lifespan_sa_37K_66sites') #change number of subjects when freesurfer issue is solved
    
    elif modality.lower() == 'fc':
        #functional connectome
        df_tr = pd.read_csv(os.path.join(data_dir,'connectomes','lifespan_controls_yeo17_tr.csv')) 
        df_te = pd.read_csv(os.path.join(data_dir,'connectomes','lifespan_controls_yeo17_te.csv'))
        df_tr.set_index('sub_id', inplace=True)
        df_te.set_index('sub_id', inplace=True)
        with open(os.path.join(root_dir,'docs','phenotypes_yeo17.txt')) as f:
            response_variables = f.read().splitlines()
        out_dir = os.path.join(root_dir,'models', model_type +'_lifespan_fc_yeo17_21K_40sites')

    elif modality.lower() == 'fa':
        # diffusion (FA)
        df_tr = pd.read_pickle(os.path.join(data_dir,'FA_clean_train.pkl')) 
        df_te = pd.read_pickle(os.path.join(data_dir,'FA_clean_test.pkl'))
        # fix some coding errors
        df_tr.columns = df_tr.columns.str.replace(" - ", "-")
        df_tr.columns = df_tr.columns.str.replace(" ", "_")
        df_te.columns = df_te.columns.str.replace(" - ", "-")
        df_te.columns = df_te.columns.str.replace(" ", "_")
        df_te = df_te.loc[df_te['site'] != 'DHCP_Evelina']
        df_tr = df_tr.loc[df_tr['site'] != 'DHCP_Evelina']
        df_tr['age'] = pd.to_numeric(df_tr['age'])
        df_te['age'] = pd.to_numeric(df_te['age'])
        with open(os.path.join(root_dir,'docs','phenotypes_fa.txt')) as f:
            idp_ids_fa = f.read().splitlines()
        response_variables = idp_ids_fa 
        out_dir = os.path.join(root_dir,'models', model_type + '_lifespan_FA_24K_19sites')

    # concatenate (split later using ptk routines)
    df = pd.concat((df_tr, df_te))

    # recode sex and add subject id
    df['sex'] = df.apply(lambda x: {0: "F", 1: "M"}[x['sex']], axis=1)
    df['sub_id'] = df.index.astype(str)

    return df, response_variables, out_dir

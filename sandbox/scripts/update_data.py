import pandas as pd
import os
import geopandas as gpd
from string_matching import string_matching

pd.set_option('display.max_rows', None)


PATH = os.path.join('../data/add_data', 'tassi_occupazione.csv')
work_rates = pd.read_csv(PATH)
df_eda = gpd.read_file('../data/merged_ecec_province.geojson')


work_rates.head() # data are in "long" format kinda
work_rates = work_rates[['Territorio','Sesso','Osservazione']] # only relevant columns

work_rates = work_rates.pivot_table( # from long to wide format
    index='Territorio', 
    columns='Sesso', 
    values='Osservazione'
    ).reset_index().rename_axis(columns=None)

work_rates['Territorio'].nunique() # 132 instad of 107 provinces
# we have more observations than expected, and some weird names

# check differences between our data and the new one
prov_eda = df_eda['prov_name'].unique()
prov_rates = work_rates['Territorio'].unique()

# match the province names
matching_provinces = string_matching(prov_rates, prov_eda, 90)


work_rates['clean_province'] = work_rates['Territorio'].map(matching_provinces)

work_rates # check results 

# Both Reggio di Calabria and Calabria are passed as Calabria
# Friuli Venezia Giulia is passed as Venezia
# Sardegna passato as Sud Sardegna
# Emilia Romagna passed as Roma

# Remove problematic entries
remove = ['Sardegna','Friuli-Venezia Giulia', 'Calabria', 'Emilia-Romagna']
work_rates = work_rates[~work_rates['Territorio'].isin(remove)]
work_rates = work_rates.dropna(subset=['clean_province'])

# clean rates dataset
work_rates = work_rates.drop('Territorio', axis=1)
work_rates = work_rates.rename(columns={'Maschi': 'wr_men', 
                                        'Femmine': 'wr_women',
                                        'Totale' : 'wr_tot'})
                                    
# merge with EDA dataset
merged_df = df_eda.merge(work_rates, left_on='prov_name', right_on='clean_province', how='left')

### ADD OTHER DATA
# ECEC diffusion
ecec_diffusion = pd.read_excel('../data/add_data/additional_health_data.xls', header=7, sheet_name='Ind. 142_P')
ecec_diffusion.rename(columns={'Unnamed: 1':"prov"}, inplace=True)
ecec_diffusion = ecec_diffusion[['prov', 2022]]

ecec_diffusion.dropna(inplace=True)

health_match = string_matching(list(ecec_diffusion['prov']), list(df_eda['prov_name']), 90)
ecec_diffusion['prov'] = ecec_diffusion['prov'].map(health_match)
ecec_diffusion.dropna(inplace=True)
ecec_diffusion.rename(columns={2022: 'ecec_diffusion'}, inplace=True)


merged_df = merged_df.merge(ecec_diffusion, left_on='prov_name', right_on='prov', how='left')

merged_df.drop(['prov_y','clean_province'], axis=1, inplace=True)

merged_df.columns
merged_df.empl_gap = (merged_df.wr_women / merged_df.wr_men).round(1)

#merged_df.to_file('../data/updated_data.geojson', driver='GeoJSON')

# transform ecec diffusion to numeric -> Bolzano is NA
merged_df['ecec_diffusion'] = pd.to_numeric(merged_df['ecec_diffusion'], errors='coerce')
merged_df.set_index('prov_name', inplace=True)
updated_csv = merged_df.select_dtypes(include=['number'])

# export CSV
#updated_csv.to_csv('../data/updated_data.csv', index=True)

# duplicate columns
prob_columns = ['wr_tot','wr_women']
updated_csv[prob_columns]
updated_csv
updated_csv.drop(prob_columns, axis=1, inplace=True)

updated_csv['empl_gap'] = (updated_csv['fem_empl_rate'] / updated_csv['wr_men']).round(2)
updated_csv['empl_gap']

updated_csv[['fem_empl_rate','wr_men','empl_gap']]
updated_csv.columns
updated_csv.to_csv('../data/updated_data.csv', index=True)


gdf = gpd.read_file('../data/updated_data.geojson')

extra_cols = set(gdf.columns) - set(updated_csv.columns)
useless_cols = ['rip_code', 'prov_area_code', 'prov_code','prov_type','prov_x', 'reg_code', 'rip_code','wr_women', 'wr_tot']
gdf.drop(useless_cols, axis=1, inplace=True)
gdf['empl_gap'] = (gdf['fem_empl_rate'] / gdf['wr_men']).round(2)

gdf.drop(['rip_code','reg_code','prov_code','prov_area_code','prov_type'], axis=1, inplace=True)
gdf['empl_gap'] = (gdf['fem_empl_rate'] / gdf['mal_empl_rate']).round(2)

gdf.to_file('../data/updated_data.geojson', driver='GeoJSON')

### ADDUTION: INACTIVITY RATES
inactivity = pd.read_csv('../data/add_data/inactivity_rates.csv')
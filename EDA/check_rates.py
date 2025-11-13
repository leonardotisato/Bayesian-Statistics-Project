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

# merge fuzzy strings


# check differences between our data and the new one
prov_eda = df_eda['prov_name'].unique()
prov_rates = work_rates['Territorio'].unique()

# match the province names
matching_provinces = string_matching(prov_rates, prov_eda, 90)

# --- 3. Apply the Mapping ---

# --- FIX 2 (The Critical One) ---
# Create a new column 'clean_province' in the work_rates DataFrame
work_rates['clean_province'] = work_rates['Territorio'].map(matching_provinces)

# Print the work_rates DataFrame to see the new column
work_rates

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



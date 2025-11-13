import pandas as pd
import os
import geopandas as gpd
PATH = os.path.join('data/add_data', 'tassi_occupazione.csv')
work_rates = pd.read_csv(PATH)

work_rates.head()
work_rates = work_rates[['Territorio','Sesso','Osservazione']]

work_rates['Territorio'].unique()

to_remove = ['Italia', 'Nord', 'Nord-ovest', 'Nord-est', 'Centro', 'Sud', 'Isole', ['Piemonte',
 '\'Valle d"\'Aosta / Vallée d"\'Aoste\'',
 'Liguria',
 'Lombardia',
 'Trentino Alto Adige / Südtirol',
 'Provincia Autonoma Bolzano / Bozen',
 'Provincia Autonoma Trento',
 'Veneto',
 'Friuli-Venezia Giulia',
 'Emilia-Romagna',
 '\'Reggio nell"\'Emilia\'',
 'Forlì-Cesena',
 'Toscana',
 'Massa-Carrara',
 'Umbria',
 'Marche',
 'Lazio',
 'Mezzogiorno',
 'Abruzzo',
 '\'L"\'Aquila\'',
 'Molise',
 'Campania',
 'Puglia',
 'Basilicata',
 'Calabria',
 'Sicilia',
 'Sardegna']]
work_rates = work_rates[~work_rates['Territorio'].isin(to_remove)]
work_rates.head()

df = pd.DataFrame()
df['prov'] = work_rates['Territorio'].unique()
df['men_rate'] = 
df.head()

pd.set_option('display.max_rows', None)

df_eda = gpd.read_file('data/merged_ecec_province.geojson')

provs_eda = [prov for prov in df_eda['prov_name'].unique()]

provs_extra = [prov for prov in df['prov'].unique() if prov not in provs_eda]


df = df[~df['prov'].isin(to_remove)]
df.shape

sorted(df['prov'].unique())
sorted(df_eda['prov_name'].unique())
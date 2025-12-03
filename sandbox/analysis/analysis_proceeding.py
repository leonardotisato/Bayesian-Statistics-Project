# Goal: use stan to model the response as a mixture of normals

# imports
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = gpd.read_file("../data/updated_data.geojson")

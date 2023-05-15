import math
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import rasterio
import rioxarray
import xarray
from netCDF4 import Dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def extract_water_mask(bqa):
    bqa_binary = np.binary_repr(bqa, width=16)
    
    # Initialize the water and cloud masks as arrays of zeros with the same shape as the BQA band
    water_mask = np.zeros(bqa.shape, dtype=bool)
    cloud_mask = np.zeros(bqa.shape, dtype=bool)
    
    # Loop through each pixel in the BQA band
    for i in range(bqa.shape[0]):
        bqa_value = bqa_binary[i]
        # Check if the first two bits of the BQA band indicate a clear water pixel
        if bqa_value[14:16] == '01':
            water_mask[i] = True
        # Check if the first two bits of the BQA band indicate a cloud or cloud shadow pixel
        elif bqa_value[14:16] in ['10', '11']:
            cloud_mask[i] = True
    
    return (water_mask,cloud_mask)

f = '../data/level_3/pixEx_JAN272022_DATA_YSI_EXO.csv'
colocated_chlorophyll_df = pd.read_csv(f, sep=',', low_memory=False,usecols=['NDCI', 'Chlorophyll ug/L']).dropna()
#colocated_chlorophyll_df = colocated_chlorophyll_df[colocated_chlorophyll_df['Chlorophyll ug/L'] > 0]
print(colocated_chlorophyll_df['Chlorophyll ug/L'].count())
print(colocated_chlorophyll_df['Chlorophyll ug/L'].max())
print(colocated_chlorophyll_df['Chlorophyll ug/L'].min())
print(colocated_chlorophyll_df['Chlorophyll ug/L'].mean())
colocated_chlorophyll_df

x = np.array(colocated_chlorophyll_df['NDCI']).reshape(-1, 1)
y = np.array(colocated_chlorophyll_df['Chlorophyll ug/L'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.1)

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
print(regr.score(x_test,y_test))

# Plot outputs
#plt.scatter(x_test, y_test, color="black")
#plt.plot(x_test, y_pred, color="red", linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()

SBC_regr = regr

file_name = 'S2B_MSI_2022_04_25_16_06_07_T17RNJ_L2R'
xr = rioxarray.open_rasterio('../data/level_2/remote_sensing/sentinel_2/'+file_name+'.nc', masked=True)

#xr.attrs['units'] = 'mm'
#print(xr.dims)
#print(xr.attrs)
#print(xr.coords)

#band442 = xr.rhos_442.sel(band=1)
#band442
#band442.plot()

#band443 = xr.rhos_443.sel(band=1)
#band443
#band443.plot()

#print(xr.dims)
#print(xr.attrs)
#print(xr.coords)

#plt.show()

df = xr.to_dataframe()

#df['NDCI'] = (df['B11'] - df['B08']).div((df['B11'] + df['B08']))
#if B8 > 0.05 then NaN else 1
df['Landmask'] = np.where(df['rhos_833'] > 0.05, 0, 1)
#df = df.loc[(df['Landmask']== 1)]
#df = df.dropna()
#log(1000 * B3_DOS)/ log(1000 * B2_DOS)
df['B3B2'] = (np.log(1000*math.pi*df['rhos_559'])).div((np.log(1000*math.pi*df['rhos_492']))).replace([np.NaN,np.inf, -np.inf], 0)
#df['SBD'] = SBD_regr.predict(df['B3B2'].values.reshape(-1,1))
df['NDCI'] = (df['rhos_704'] - df['rhos_665']) / (df['rhos_704'] + df['rhos_665'])
df['SCD'] = SBC_regr.predict(df['NDCI'].fillna('0',inplace=False).values.reshape(-1,1))
print(df['SCD'].min())
df

ocean_surface_df = df.loc[(df['Landmask']== 1)]
figure = plt.figure(figsize=(10,10))
#figure = plt.figure()
ax = figure.add_subplot(projection='3d')
ax.scatter(ocean_surface_df['lon'], ocean_surface_df['lat'], ocean_surface_df['SCD'])
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Satellite Derived Chlorophyll')
plt.show()
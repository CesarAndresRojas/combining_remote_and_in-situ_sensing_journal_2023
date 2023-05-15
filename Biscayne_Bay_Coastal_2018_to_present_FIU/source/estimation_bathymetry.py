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

f = '../data/level_3/pixEx_JAN272022_DATA_SONTEK_M9.csv'
colocated_depth_df = pd.read_csv(f, sep=',', low_memory=False,usecols=['B3B2', 'Total Water Column (m)']).dropna()
print(colocated_depth_df['Total Water Column (m)'].max())
print(colocated_depth_df['Total Water Column (m)'].min())
print(colocated_depth_df['Total Water Column (m)'].mean())
colocated_depth_df

x = np.array(colocated_depth_df['B3B2']).reshape(-1, 1)
y = np.array(colocated_depth_df['Total Water Column (m)'])

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

plt.show()

SBD_regr = regr

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
df['SBD'] = SBD_regr.predict(df['B3B2'].values.reshape(-1,1))
#df['NDCI'] = (df['rhos_704'] - df['rhos_665']) / (df['rhos_704'] + df['rhos_665'])
#df['SCD'] = SBC_regr.predict(df['NDCI'].fillna('0',inplace=False).values.reshape(-1,1))
print(df['SBD'].min())

ocean_surface_df = df.loc[(df['Landmask']== 1)]
figure = plt.figure(figsize=(10,10))
#figure = plt.figure()
ax = figure.add_subplot(projection='3d')
#ax = figure.add_subplot()
sc = ax.scatter(ocean_surface_df['lon'], ocean_surface_df['lat'],ocean_surface_df['SBD'], c= ocean_surface_df['SBD'], cmap="copper")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
#ax.set_zlabel('Satellite Derived Bathymetry')
plt.colorbar(sc)
plt.show()
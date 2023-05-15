import cv2
import datetime
import json
import math
import os

from netCDF4 import Dataset
from scipy.interpolate import griddata
from shapely.geometry import Polygon, shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, PolynomialFeatures
from sklearn.svm import SVR
import matplotlib

import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import water_parameter_ranges as constants
import xarray

file_path = '../data/level_4/'

matplotlib.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset':'cm'})
# figsize=(10, 7.5)
# figsize=(3.5, 2.5)
# figsize = (7.16,2.5)


def create_knn_regression_model(x, y, min_x, max_x, plot_graph=False, x_label='x', y_label='y', plot_title='KNN Regression', n_neighbors=1, file_prefix=''):
    # Fit the KNN regression model
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(x, y)

    # Make predictions using the same dataset
    y_pred = knn.predict(x)

    # Print the results
    print(y_label)
    print('Mean squared error: %.2f' % mean_squared_error(y, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y, y_pred))

    if plot_graph:
        # Set font family to Arial
        #plt.rcParams['font.family'] = 'Arial'
        # Predict the output for new input data
        x_plot = np.linspace(min_x, max_x, 100).reshape((-1, 1))
        y_pred_plot = knn.predict(x_plot)

        # Create a new figure and axes
        fig, ax = plt.subplots()

        # Plot the data and the regression line
        ax.scatter(x, y, color='k')
        ax.plot(x_plot, y_pred_plot, color='k')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)

        plt.savefig(file_path + file_prefix + '_knn_regression.pdf', bbox_inches='tight')
        plt.savefig(file_path + file_prefix + '_knn_regression.png', bbox_inches='tight')
        plt.savefig(file_path + file_prefix + '_knn_regression.svg', bbox_inches='tight')

        plt.show()

    return knn


def create_knn_regression_models(water_column_f, neighbors_list=[5, 5, 5, 5], plot_graph_list=[False, False, False, False], file_prefix=''):
    water_column_df = pd.read_csv(water_column_f, sep=',', low_memory=False).dropna()

    # Find index of max depth and min depth greater than 0
    max_depth_idx = water_column_df['Depth m'].idxmax()
    min_depth_idx = water_column_df.loc[water_column_df['Depth m'] > 0, 'Depth m'].idxmin()

    # Filter the dataframe based on max depth and positive depths
    water_column_df_filtered = water_column_df.loc[:max_depth_idx].loc[water_column_df['Depth m'] > 0]

    lat = water_column_df_filtered['GPS Latitude °'].iloc[0]
    lon = water_column_df_filtered['GPS Longitude °'].iloc[0]

    # Extract the columns of interest
    x = water_column_df_filtered['Depth m'].values.reshape(-1, 1)    
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    
    file_name = os.path.splitext(os.path.basename(water_column_f))[0]
    
    y = water_column_df_filtered['ODO mg/L'].values
    SDoD_knn = hlp.create_knn_regression_model(x_scaled, y, 0, 1, plot_graph_list[0], 'Normalized Depth', 'ODO mg/L', file_name, neighbors_list[0], file_prefix + 'do')
    
    y = water_column_df_filtered['Chlorophyll ug/L'].values
    SCD_knn = hlp.create_knn_regression_model(x_scaled, y, 0, 1, plot_graph_list[1], 'Normalized Depth', 'Chlorophyll ug/L', file_name, neighbors_list[1], file_prefix + 'chla')
    
    y = water_column_df_filtered['Temp °C'].values
    STempD_knn = hlp.create_knn_regression_model(x_scaled, y, 0, 1, plot_graph_list[2], 'Normalized Depth', 'Temp °C', file_name, neighbors_list[2], file_prefix + 'temp')
    
    y = water_column_df_filtered['Turbidity FNU'].values
    STurbD_knn = hlp.create_knn_regression_model(x_scaled, y, 0, 1, plot_graph_list[3], 'Normalized Depth', 'Turbidity FNU', file_name, neighbors_list[3], file_prefix + 'turb')
    
    return lat, lon, SDoD_knn, SCD_knn, STempD_knn, STurbD_knn


def create_linear_regression_model(x, y, min_x, max_x, plot_graph=False, x_label='x', y_label='y', plot_title='Linear Regression', file_prefix=''):
    # Fit the linear regression model
    regr = LinearRegression()
    regr.fit(x, y)

    # Make predictions using the same dataset
    y_pred = regr.predict(x)

    # Print the results
    print(y_label)
    print('Coefficients: \n', regr.coef_)
    print('Mean squared error: %.2f' % mean_squared_error(y, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y, y_pred))

    if plot_graph:
        # Set font family to Arial
        #plt.rcParams['font.family'] = 'Arial'
        # Predict the output for new input data
        x_plot = np.linspace(min_x, max_x, 100).reshape((-1, 1))
        y_pred_plot = regr.predict(x_plot)

        # Create a new figure and axes
        fig, ax = plt.subplots()
        
        # Plot the data and the regression line
        ax.scatter(x, y, color='k')
        ax.plot(x_plot, y_pred_plot, color='k')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)

        plt.savefig(file_path + file_prefix + '_linear_regression.pdf', bbox_inches='tight')
        plt.savefig(file_path + file_prefix + '_linear_regression.png', bbox_inches='tight')
        plt.savefig(file_path + file_prefix + '_linear_regression.svg', bbox_inches='tight')

        plt.show()

    return regr


def create_2d_linear_regression_model(x, y, plot_graph=False, x_label='x', y_label='y', plot_title='Linear Regression', file_prefix=''):
    # Fit the linear regression model
    regr = LinearRegression()
    regr.fit(x, y)

    # Make predictions using the same dataset
    y_pred = regr.predict(x)

    # Print the results
    print(y_label)
    print('Coefficients: \n', regr.coef_)
    print('Mean squared error: %.2f' % mean_squared_error(y, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y, y_pred))

    if plot_graph:
        # Set font family to Arial
        #plt.rcParams['font.family'] = 'Arial'
        # Predict the output for new input data
        x1 = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
        x2 = np.linspace(min(x[:, 1]), max(x[:, 1]), 100)
        xx1, xx2 = np.meshgrid(x1, x2)
        zz = np.zeros(xx1.shape)
        for i in range(xx1.shape[0]):
            for j in range(xx1.shape[1]):
                xx = np.array([xx1[i, j], xx2[i, j]]).reshape((1, -1))
                zz[i, j] = regr.predict(xx)

        # Plot the data and the regression surface
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:, 0], x[:, 1], y, color='k')
        ax.plot_surface(xx1, xx2, zz, color=(0.5, 0.5, 0.5, 0.5))
        ax.set_xlabel(x_label[0])
        ax.set_ylabel(x_label[1])
        ax.set_zlabel(y_label, labelpad=1)
        ax.zaxis.set_major_formatter('{:.01f}'.format)
        ax.set_title(plot_title, pad=0)

        plt.savefig(file_path + file_prefix + '_2D_linear_regression.pdf')
        plt.savefig(file_path + file_prefix + '_2D_linear_regression.png')
        plt.savefig(file_path + file_prefix + '_2D_linear_regression.svg')

        plt.show()

    return regr


def create_polynomial_regression_model(x, y, min_x, max_x, degree=2, plot_graph=False, x_label='x', y_label='y', plot_title='Polynomial Linear Regression', file_prefix=''):
    # Create polynomial features and fit the linear regression model
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)
    regr = LinearRegression()
    regr.fit(x_poly, y)

    # Make predictions using the same dataset
    y_pred = regr.predict(x_poly)

    # Print the results
    print(y_label)
    print('Coefficients: \n', regr.coef_)
    print('Mean squared error: %.2f' % mean_squared_error(y, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y, y_pred))

    if plot_graph:
        # Set font family to Arial
        #plt.rcParams['font.family'] = 'Arial'
        # Predict the output for new input data
        x_plot = np.linspace(min_x, max_x, 100).reshape((-1, 1))
        x_plot_poly = poly_features.transform(x_plot)
        y_pred_plot = regr.predict(x_plot_poly)

        # Plot the data and the polynomial regression curve
        plt.scatter(x, y)
        plt.plot(x_plot, y_pred_plot, color='r')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)

        plt.savefig(file_path + file_prefix + '_polynomial_regression.pdf', bbox_inches='tight')
        plt.savefig(file_path + file_prefix + '_polynomial_regression.png', bbox_inches='tight')
        plt.savefig(file_path + file_prefix + '_polynomial_regression.svg', bbox_inches='tight')

        plt.show()

    return poly_features, regr


def create_polynomial_regression_model_train_test_split(x, y, min_x, max_x, degree=2, plot_graph=False, x_label='x', y_label='y', plot_title='Polynomial Linear Regression'):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # Create polynomial features and fit the linear regression model
    poly_features = PolynomialFeatures(degree=degree)
    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.fit_transform(x_test)
    regr = LinearRegression()
    regr.fit(x_train_poly, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test_poly)

    # Print the results
    print(y_label)
    print('Coefficients: \n', regr.coef_)
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    print(regr.score(x_test_poly, y_test))

    # Create a polynomial linear regression model with an intercept
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)

    regr = LinearRegression(fit_intercept=True)

    # Fit the model to the data
    regr.fit(x_poly, y)

    if plot_graph:
        # Set font family to Arial
        #plt.rcParams['font.family'] = 'Arial'
        # Predict the output for new input data
        x_test = np.linspace(min_x, max_x, 100).reshape((-1, 1))
        x_test_poly = poly_features.transform(x_test)
        y_pred = regr.predict(x_test_poly)

        # Plot the data and the polynomial regression curve
        plt.scatter(x, y)
        plt.plot(x_test, y_pred, color='r')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)
        plt.show()

    return poly_features, regr


def create_polynomial_regression_models(water_column_f, degree_list=[2, 2, 2, 2], plot_graph_list=[False, False, False, False], file_prefix=''):
    water_column_df = pd.read_csv(water_column_f, sep=',', low_memory=False).dropna()
    # water_column_df_filtered['surface chla'] = water_column_df.loc[min_depth_idx]['Chlorophyll ug/L'].min()
    # x = water_column_df_filtered[['Depth m', 'surface chla']]

    # Find index of max depth and min depth greater than 0
    max_depth_idx = water_column_df['Depth m'].idxmax()
    min_depth_idx = water_column_df.loc[water_column_df['Depth m'] > 0, 'Depth m'].idxmin()

    # Filter the dataframe based on max depth and positive depths
    water_column_df_filtered = water_column_df.loc[:max_depth_idx].loc[water_column_df['Depth m'] > 0]

    lat = water_column_df_filtered['GPS Latitude °'].iloc[0]
    lon = water_column_df_filtered['GPS Longitude °'].iloc[0]

    # Extract the columns of interest
    x = water_column_df_filtered['Depth m'].values.reshape(-1, 1)    
    # y = water_column_df_filtered['Chlorophyll ug/L'].values
    # Adjust x and y values to start at 0
    # min_depth = water_column_df.loc[min_depth_idx, 'Depth m']
    # surface_chla = water_column_df.loc[min_depth_idx, 'Chlorophyll ug/L']
    # x_adjusted = x - min_depth
    # y_adjusted = y - surface_chla
    # Scale the training set and labels
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    # x_scaled = scaler.fit_transform(x_adjusted)
    # y_scaled = scaler.fit_transform(y_adjusted.reshape(-1, 1))

    file_name = os.path.splitext(os.path.basename(water_column_f))[0]

    y = water_column_df_filtered['ODO mg/L'].values
    SDoD_poly, SDoD_regr = hlp.create_polynomial_regression_model(x_scaled, y, 0, 1, degree_list[0], plot_graph_list[0], 'Normalized Depth', 'ODO mg/L', file_name, file_prefix + 'do')

    # Create a polynomial features transformer for model
    y = water_column_df_filtered['Chlorophyll ug/L'].values
    SCD_poly, SCD_regr = hlp.create_polynomial_regression_model(x_scaled, y, 0, 1, degree_list[1], plot_graph_list[1], 'Normalized Depth', 'Chlorophyll ug/L', file_name, file_prefix + 'chla')

    y = water_column_df_filtered['Temp °C'].values
    STempD_poly, STempD_regr = hlp.create_polynomial_regression_model(x_scaled, y, 0, 1, degree_list[2], plot_graph_list[2], 'Normalized Depth', 'Temp °C', file_name, file_prefix + 'temp')

    y = water_column_df_filtered['Turbidity FNU'].values
    STurbD_poly, STurbD_regr = hlp.create_polynomial_regression_model(x_scaled, y, 0, 1, degree_list[3], plot_graph_list[3], 'Normalized Depth', 'Turbidity FNU', file_name, file_prefix + 'turb')

    return lat, lon, SDoD_poly, SDoD_regr, SCD_poly, SCD_regr, STempD_poly, STempD_regr, STurbD_poly, STurbD_regr


def haversine(lat1, lon1, lat2, lon2):
    '''
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    '''
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_closest_model(lat, lon, models):
    min_dist = float('inf')
    closest_model = None
    for model in models:
        model_lat, model_lon = model[0], model[1]
        dist = haversine(lat, lon, model_lat, model_lon)
        if dist < min_dist:
            min_dist = dist
            closest_model = model
    if closest_model is None:
        return None
    else:
        # lat, lon, SDoD_poly, SDoD_regr, SCD_poly, SCD_regr, STempD_poly, STempD_regr, STurbD_poly, STurbD_regr = closest_model
        # return lat, lon, SDoD_poly, SDoD_regr, SCD_poly, SCD_regr, STempD_poly, STempD_regr, STurbD_poly, STurbD_regr
        return closest_model


def plot_3d_scatter_with_colorbar(x, y, z, z_label, plot_title, cmap='viridis', file_prefix=''):
    # Set font family to Arial
    #plt.rcParams['font.family'] = 'Arial'
    # Create plot with color bar
    fig = plt.figure()
    # fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Create a scatter plot with a colormap
    sc = ax.scatter(x, y, z, c=z, cmap=cmap)

    # Set the labels for the axes
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel(z_label)

    # Add a colorbar to the plot
    # Add a color bar which maps values to colors.
    # cbar = fig.colorbar(sc, orientation='horizontal', shrink=0.60, aspect=10, location='bottom', pad=-0.01)
    cbar = fig.colorbar(sc, location='top', shrink=0.6, pad=-0.15, aspect=20, orientation='horizontal')
    # cbar.ax.tick_params(labelsize=7)

    # Set the label for the colorbar
    cbar.ax.set_xlabel(z_label)

    # Show the plot
    plt.title(plot_title)

    plt.savefig(file_path + file_prefix + '_3D_Scatter.pdf')  # , bbox_inches='tight')
    plt.savefig(file_path + file_prefix + '_3D_Scatter.png')  # , bbox_inches='tight')
    plt.savefig(file_path + file_prefix + '_3D_Scatter.svg')

    plt.show()


def plot_2d_scatter_with_colorbar(x, y, z, z_label, plot_title, cmap='viridis', file_prefix='', marker_size=5):
    # Set font family to Arial
    #plt.rcParams['font.family'] = 'Arial'

    # Create a scatter plot with a colormap
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, cmap=cmap, marker='s', s=marker_size)

    # Set the labels for the axes
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Add a colorbar to the plot
    cbar = fig.colorbar(sc, location='right', shrink=0.9, pad=0.01, aspect=20, orientation='vertical')
    cbar.ax.set_ylabel(z_label)

    # Show the plot
    plt.title(plot_title)

    plt.savefig(file_path + file_prefix + '_2D_Scatter.pdf', bbox_inches='tight')
    plt.savefig(file_path + file_prefix + '_2D_Scatter.png', bbox_inches='tight')
    plt.savefig(file_path + file_prefix + '_2D_Scatter.svg', bbox_inches='tight')

    plt.show()


# This creates the green customized colormap in the format RGBA
def green_colormap(num_colors, alpha=0.9):
    start_color = np.array([[1 / 255, 127 / 255, 0 / 255, alpha ]])  # Forest green
    end_color = np.array([[211 / 255, 241 / 255, 210 / 255, alpha ]])  # Almost white
    t = np.linspace(0, 1, num_colors)
    colors = np.zeros((num_colors + 1, 4))
    colors[0] = np.array([[198 / 255, 172 / 255, 167 / 255, alpha * 0.6 ]])  # Brown or cinnamon
    for i in range(num_colors):
        colors[i + 1] = t[i] * start_color + (1 - t[i]) * end_color
    
    return colors

def create_colormap(num_colors, 
                    high_concentration_color_or_cmap, 
                    low_concentration_color = (250/255,250/255,250/255,0.9),
                    land_color = (198 / 255, 172 / 255, 167 / 255, 0.6 ) ):
    
    '''
    num_colors: how many bins the colorbar has
    land_color: color of the land
    two options:
    
    1) high_concentration_color_or_cmap and low_concentration_color are tuples or 
    lists with four numbers of colors in normalized RGBA format e.g. white = (1.0,1.0,1.0,1.0)
    
    2) Import the module cm of matplotlib, in that case high_concentration_color_or_cmap is a 
    matplotlib.colors.LinearSegmentedColormap object e.g. cm.hot
    
    
    '''
    
    if (type(high_concentration_color_or_cmap)== tuple or type(high_concentration_color_or_cmap)==list):
        high_concentration_color = np.array( list(high_concentration_color_or_cmap))  
        low_concentration_color = np.array( list(low_concentration_color))  
        land_color = np.array( list(land_color))

        t = np.linspace(0, 1, num_colors).reshape(-1,1)

        # convex combination (1-t)*low_color +t*high_color

        colors = t*(high_concentration_color-low_concentration_color) + low_concentration_color
        
    else:
        colors = high_concentration_color_or_cmap(np.linspace(0,256,num_colors, dtype =int))
    colors = np.vstack((land_color, colors))
    return colors


def plot_sdue(data, vmin, vmax, z_limit, plot_title, color_bar_label, file_prefix=''):
    # Set font family to Arial
    #plt.rcParams['font.family'] = 'Arial'
    # replace 0 values with -1
    data = np.where(data == 0, np.nan, data)
    # create the 3D plot
    x, y, z = data.shape
    X, Y, Z = np.meshgrid(np.arange(x), np.arange(y), np.arange(z) * -1, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    cmap = plt.get_cmap('magma')
    # cmap.set_extremes(bad=bad, under=under, over=over)
    # cmap.set_bad('gray')
    cmap.set_over('red')
    cmap.set_under('blue')

    # norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    # mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    surfs = []
    for i in range(z_limit):
        # facecolors = mapper.to_rgba(data[:,:,i])
        # facecolors = cmap(data[:,:,i]/vmax)
        facecolors = cmap((data[:,:, i] - vmin) / (vmax - vmin))
        surf = ax.plot_surface(X[:,:, i], Y[:,:, i], Z[:,:, i], cmap=cmap, facecolors=facecolors, linewidth=0, antialiased=False)
        surf.set_clim(vmin=vmin, vmax=vmax)
        surfs.append(surf)
    # add a colorbar
    cbar = fig.colorbar(surfs[0], location='top', shrink=0.6, pad=-0.15, aspect=20, orientation='horizontal', extend='both')

    # Set the label for the colorbar
    cbar.ax.set_xlabel(color_bar_label)

    # Get the colorbar object
    # cb_obj = plt.getp(cbar.ax.axes, 'xticklabels')

    # Set the first tick to 'Land' and the color to white
    # cb_obj[0].set_text('Land')

    # Update the colorbar
    # cbar.ax.axes.set_xticklabels(cb_obj)

    # set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    plt.title(plot_title)
    plt.savefig(file_path + file_prefix + '_surface_SDUE.pdf')#, bbox_inches='tight')
    plt.savefig(file_path + file_prefix + '_surface_SDUE.png')#, bbox_inches='tight')
    plt.savefig(file_path + file_prefix + '_surface_SDUE.svg')#, bbox_inches='tight')
    plt.show()


def plot_sdue_contourf(data, plot_title, color_bar_label,
                       file_prefix='', high_concentration_color_or_cmap =(0,255/255,0,1.0),
                      low_concentration_color = (240/255,240/255,240/255,1.0) ,
                      n_layers = 4,
                      show_tick_labels = False):
    
    # File is given in x,y,z format

    mask = (data[:,:,0]!=0.0)
        

    data = np.swapaxes(data, -1,0)
    mask = np.swapaxes(mask,0,-1)

    z_length, y_length, x_length = data.shape 
        

    # This creates the mesh grid
    
    x, y = np.linspace(0, x_length, x_length), np.linspace(0, y_length, y_length)
    XX, YY = np.meshgrid(x, y)

    # Masking the layers

    n_layers = 4
    num_cols = 10
    percentiles = np.linspace(0, 100, num_cols)
    
    mask = np.tile(mask, (n_layers,1,1))
    masked_data = data[: n_layers][mask * (data[: n_layers] > 0)]
    

    
    Levels = np.unique(np.percentile(masked_data, percentiles))
    Levels = np.append([-1], Levels)
    
    colors = create_colormap(Levels.shape[0], high_concentration_color_or_cmap,
                             (240/255,240/255,240/255,1.0) )

    # Graphics part
    
    num_xticks, num_yticks, num_zticks = 5, 5, n_layers
    
    fig = plt.figure(figsize = (9,9))
    ax = plt.subplot(projection='3d')

    for i in range(0, n_layers):
        current_layer = data[i]
        plot3d = ax.contourf(XX, YY, current_layer, levels= Levels , colors=colors, offset=-i)

    
    # Setting x,y and z ticks
    
    ax.set_yticks(np.linspace(0, y_length, num_xticks))
    ax.set_yticklabels(
        np.round(np.linspace(29.9020, 25.9160, num_xticks), decimals = 3))
    ax.set_xticks(np.linspace(0, x_length, num_yticks))
    ax.set_xticklabels(
        np.round(np.linspace(-80.40, -80.30, num_yticks), decimals = 3))
    ax.set_zticks(np.linspace(0, -n_layers + 1, n_layers))
    ax.set_zticklabels(np.linspace(0, n_layers - 1, n_layers).astype(int))

    # x,y,z axis labels
    labelpad = 10
    
    if not show_tick_labels:
        labelpad = -15
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
    ax.set_xlabel('Longitude', labelpad=labelpad)
    ax.set_ylabel('Latitude', labelpad=labelpad)
    ax.set_zlabel('Depth', labelpad=labelpad)
    ax.set_zlim(-n_layers, 0)

    cbar = fig.colorbar(plot3d, location='top', shrink=0.6, pad=-0.15, aspect=15, orientation='horizontal')
    cbar.ax.set_xlabel(color_bar_label)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks(Levels )
    
    # Round the tick labels to 2 decimal places
    
    tick_labels = np.round(Levels, decimals=2).tolist()

    # Add land label to the colorbar
    tick_labels[0] = 'Land'
    cbar.set_ticklabels(tick_labels)
    plt.title(plot_title)
    plt.savefig(file_path + file_prefix + '_contourf_SDUE.pdf' , bbox_inches='tight')
    plt.savefig(file_path + file_prefix + '_contourf_SDUE.png' , bbox_inches='tight')
    plt.savefig(file_path + file_prefix + '_contourf_SDUE.svg' , bbox_inches='tight')
    plt.show()

def print_stats(df, col):
    print('Statistics:')
    print(f'Number of values in {col}: {len(df)}')
    print(f'Minimum value in {col}: {df[col].min()}')
    print(f'Maximum value in {col}: {df[col].max()}')
    print(f'Mean value in {col}: {df[col].mean()}')
    print(f"Number of negative values in {col}: {(df[col] < 0).sum()}")
    print('*'*10)


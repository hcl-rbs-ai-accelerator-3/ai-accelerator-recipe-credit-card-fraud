# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:10:16 2019

This module contains various functions which can be helpful while doing the data 
analysis and data pre-processing. This modules includes functions for univariate 
analysis, bivariate analysis, multivariate analysis, outlier detection, missing
 value imputations, data profiling, etc.
 
 Usage : This file needs to be placed in the same folder or needs to be referenced 
 accordingly

@author: ManmohanSh
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas_profiling
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from statsmodels.stats.outliers_influence import variance_inflation_factor  

from scipy.signal import savgol_filter
from scipy.signal import lfilter 

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

#% matplotlib inline

import warnings

warnings.filterwarnings("ignore")

global_plt= plt
global_sns= sns
def univariate(column_name,plt=None,sns=None,bins=None, hist=True, kde=True, rug=False, 
               fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, 
               vertical=False, norm_hist=False, axlabel=None, label=None, ax=None):
    
    '''
    This function plots distribution graph for the one dimensional data, also combines the matplotlib hist function (with automatic calculation of a good default bin size) with the seaborn kdeplot() and rugplot() functions. It can also fit scipy.stats distributions and plot the estimated PDF over the data.

    Parameters:	
        
    column_name : Series, 1d-array, or list.
    
    plt : matplotlib object containing specific properties, optional
    
    sns : seaborn object containing specific properties, optional
    
    Observed data. If this is a Series object with a name attribute, the name will be used to label the data axis.
    
    bins : argument for matplotlib hist(), or None, optional
    
    Specification of hist bins, or None to use Freedman-Diaconis rule.
    
    hist : bool, optional
    
    Whether to plot a (normed) histogram.
    
    kde : bool, optional
    
    Whether to plot a gaussian kernel density estimate.
    
    rug : bool, optional
    
    Whether to draw a rugplot on the support axis.
    
    fit : random variable object, optional
    
    An object with fit method, returning a tuple that can be passed to a pdf method a positional arguments following an grid of values to evaluate the pdf on.
    
    {hist, kde, rug, fit}_kws : dictionaries, optional
    
    Keyword arguments for underlying plotting functions.
    
    color : matplotlib color, optional
    
    Color to plot everything but the fitted curve in.
    
    vertical : bool, optional
    
    If True, observed values are on y-axis.
    
    norm_hist : bool, optional
    
    If True, the histogram height shows a density rather than a count. This is implied if a KDE or fitted density is plotted.
    
    axlabel : string, False, or None, optional
    
    Name for the support axis label. If None, will try to get it from a.namel if False, do not set a label.
    
    label : string, optional
    
    Legend label for the relevent component of the plot
    
    ax : matplotlib axis, optional
    
    if provided, plot on this axis
    
    Returns:	
    ax : matplotlib Axes
    
    Returns the Axes object with the plot for further tweaking.
    
    Example-
    
    x = np.random.randn(100)
    univariate(x)
    
    '''  
    
    
    try:
        if plt == None:
            plt=global_plt
        if sns == None:
            sns=global_sns
        
        sns.distplot(column_name, bins, hist, kde, rug, fit, hist_kws, kde_kws, rug_kws, fit_kws, color, 
                     vertical, norm_hist, axlabel, label, ax)
    
    
    
    except Exception as e:
        print('Exception occurred in univariate() in data quality module')
        print(e)
   
    
def bivariate(type='box',x=None, y=None, hue=None, data=None,plt=None,sns=None,vars=None, x_vars=None, y_vars=None,
              kind='scatter', stat_func=None,markers=None, diag_kind='auto',aspect=1, 
              height=6, ratio=5, space=0.2,dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, 
              annot_kws=None,order=None, hue_order=None,orient=None, color=None, palette=None, saturation=0.75, 
              width=0.8, dodge=True,fliersize=5, linewidth=None,whis=1.5, notch=False, ax=None,plot_kws=None, 
              diag_kws=None, grid_kws=None, size=None,style=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, style_order=None, 
                              x_bins=None, y_bins=None, units=None, estimator=None, ci=95, n_boot=1000, alpha='auto',
                              x_jitter=None, y_jitter=None, legend='brief', dashes=True,sort=True, err_style='band', err_kws=None,**kwargs):
    
    
            
                
    
    
    
    
    
    '''
    Draws various types of bivariate plots like joint plot, box plot, pair plot or count plot depending uplon the parameter provided.
    
    Parameters:	
        
    type : type of plot to be drawn, options are box, joint, pair, count and line. box is bydefault.    
        
    x, y, hue : names of variables in data or vector data, optional
    
    Inputs for plotting long-form data. See examples for interpretation.
    
    data : DataFrame, array, or list of arrays, optional
    
    Dataset for plotting. If x and y are absent, this is interpreted as wide-form. Otherwise it is expected to be long-form.
    
    plt : matplotlib object containing specific properties, optional
    
    sns : seaborn object containing specific properties, optional
    
    vars : list of variable names, optional

    Variables within data to use, otherwise use every column with a numeric datatype.

    {x, y}_vars : lists of variable names, optional

    Variables within data to use separately for the rows and columns of the figure; i.e. to make a non-square plot.

    kind : {‘scatter’, ‘reg’}, optional

    Kind of plot for the non-identity relationships.
    
    kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional

    Kind of plot to draw.

    stat_func : callable or None, optional
	
	diag_kind : {‘auto’, ‘hist’, ‘kde’}, optional

	Kind of plot for the diagonal subplots. The default depends on whether "hue" is used or not.

	markers : single matplotlib marker code or list, optional

	Either the marker to use for all datapoints or a list of markers with a length the same as the number of levels in the hue variable so that differently colored points will also have different scatterplot markers.

	aspect : scalar, optional

	Aspect * height gives the width (in inches) of each facet.
	
	height : scalar, optional

	Height (in inches) of each facet.
	
	ratio : numeric, optional

	Ratio of joint axes height to marginal axes height.

	space : numeric, optional

	Space between the joint and marginal axes

	dropna : bool, optional

	If True, remove observations that are missing from x and y.

	{x, y}lim : two-tuples, optional

	Axis limits to set before plotting.

	{joint, marginal, annot}_kws : dicts, optional

	Additional keyword arguments for the plot components.
	
	

    order, hue_order : lists of strings, optional
    
    Order to plot the categorical levels in, otherwise the levels are inferred from the data objects.
    
    orient : “v” | “h”, optional
    
    Orientation of the plot (vertical or horizontal). This is usually inferred from the dtype of the input variables, but can be used to specify when the “categorical” variable is a numeric or when plotting wide-form data.
    
    color : matplotlib color, optional
    
    Color for all of the elements, or seed for a gradient palette.
    
    palette : palette name, list, or dict, optional
    
    Colors to use for the different levels of the hue variable. Should be something that can be interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
    
    saturation : float, optional
    
    Proportion of the original saturation to draw colors at. Large patches often look better with slightly desaturated colors, but set this to 1 if you want the plot colors to perfectly match the input color spec.
    
    width : float, optional
    
    Width of a full element when not using hue nesting, or width of all the elements for one level of the major grouping variable.
    
    dodge : bool, optional
    
    When hue nesting is used, whether elements should be shifted along the categorical axis.
    
    fliersize : float, optional
    
    Size of the markers used to indicate outlier observations.
    
    linewidth : float, optional
    
    Width of the gray lines that frame the plot elements.
    
    whis : float, optional
    
    Proportion of the IQR past the low and high quartiles to extend the plot whiskers. Points outside this range will be identified as outliers.
    
    notch : boolean, optional
    
    Whether to “notch” the box to indicate a confidence interval for the median. There are several other parameters that can control how the notches are drawn; see the plt.boxplot help for more information on them.
    
    ax : matplotlib Axes, optional
    
    Axes object to draw the plot onto, otherwise uses the current Axes.
	
	{plot, diag, grid}_kws : dicts, optional

	Dictionaries of keyword arguments.
	
	style_order : list, optional

	Specified order for appearance of the style variable levels otherwise they are determined from the data. Not relevant when the style variable is numeric.

	{x,y}_bins : lists or arrays or functions

	Currently non-functional.

	units : {long_form_var}

	Grouping variable identifying sampling units. When used, a separate line will be drawn for each unit with appropriate semantics, but no legend entry will be added. Useful for showing distribution of experimental replicates when exact identities are not needed.

	Currently non-functional.

	estimator : name of pandas method or callable or None, optional

	Method for aggregating across multiple observations of the y variable at the same x level. If None, all observations will be drawn. Currently non-functional.

	ci : int or “sd” or None, optional

	Size of the confidence interval to draw when aggregating with an estimator. “sd” means to draw the standard deviation of the data. Setting to None will skip bootstrapping. Currently non-functional.

	n_boot : int, optional

	Number of bootstraps to use for computing the confidence interval. Currently non-functional.

	alpha : float

	Proportional opacity of the points.

	{x,y}_jitter : booleans or floats

	Currently non-functional.

	legend : “brief”, “full”, or False, optional

	How to draw the legend. If “brief”, numeric hue and size variables will be represented with a sample of evenly spaced values. If “full”, every group will get an entry in the legend. If False, no legend data is added and no legend is drawn.
		
    dashes : boolean, list, or dictionary, optional

    Object determining how to draw the lines for different levels of the style variable. Setting to True will use default dash codes, or you can pass a list of dash codes or a 
    
    dictionary mapping levels of the style variable to dash codes. Setting to False will use solid lines for all subsets. Dashes are specified as in matplotlib: a tuple of 
    
    (segment, gap) lengths, or an empty string to draw a solid line.
    
    sort : boolean, optional

    If True, the data will be sorted by the x and y variables, otherwise lines will connect points in the order they appear in the dataset.

    err_style : “band” or “bars”, optional

    Whether to draw the confidence intervals with translucent error bands or discrete error bars.

    err_band : dict of keyword arguments

    Additional paramters to control the aesthetics of the error bars. The kwargs are passed either to ax.fill_between or ax.errorbar, depending on the err_style.
    
    kwargs : key, value mappings
    
    Other keyword arguments are passed through to plt.boxplot at draw time.
    
    Returns:	
    ax : matplotlib Axes
    
    Returns the Axes object with the plot drawn onto it.
    
    
    '''
    
    
    
    
    
    
    try:
        if plt==None:
            plt=global_plt
        if sns==None:
            sns=global_sns
        
        if type=='box':
            if kwargs.__len__() == 0:
                sns.boxplot(x, y, hue, data, order, hue_order, orient, color, palette, saturation, width, dodge, 
                fliersize, linewidth, whis, notch, ax)
            else:
                sns.boxplot(x, y, hue, data, order, hue_order, orient, color, palette, saturation, width, dodge, 
                fliersize, linewidth, whis, notch, ax, kwargs)
                
                
        elif type=='joint':
            if kwargs.__len__() == 0:
                sns.jointplot(x, y, data, kind, stat_func, color, height, ratio, space,dropna, xlim, ylim, 
                              joint_kws, marginal_kws, annot_kws)
            else:
                sns.jointplot(x, y, data, kind, stat_func, color, height, ratio, space,dropna, xlim, ylim, 
                              joint_kws, marginal_kws, annot_kws, kwargs)
        elif type=='pair':
            if kwargs.__len__() == 0:
                sns.pairplot(data, hue, hue_order, palette, vars, x_vars, y_vars, 
                         kind, diag_kind, markers, height, aspect, dropna, 
                         plot_kws, diag_kws, grid_kws, size)
            else:
                sns.pairplot(data, hue, hue_order, palette, vars, x_vars, y_vars, 
                         kind, diag_kind, markers, height, aspect, dropna, 
                         plot_kws, diag_kws, grid_kws, size, kwargs)
            
        elif type=='count':
            if kwargs.__len__() == 0:
                sns.countplot(x, y, hue, data, order, hue_order, orient, color, palette, saturation, 
                              dodge, ax)
            else:
                sns.countplot(x, y, hue, data, order, hue_order, orient, color, palette, saturation, 
                              dodge, ax, kwargs)
                
            
                
        elif type=='scatter':
            if kwargs.__len__() == 0:
                sns.scatterplot(x, y, hue, style, size, data, palette, hue_order, hue_norm, sizes, size_order, size_norm,
                                markers, style_order, x_bins, y_bins, units, estimator, ci, n_boot, alpha, x_jitter, y_jitter,
                                legend, ax)
            else:
                sns.scatterplot(x, y, hue, style, size, data, palette, hue_order, hue_norm, sizes, size_order, size_norm,
                                markers, style_order, x_bins, y_bins, units, estimator, ci, n_boot, alpha, x_jitter, y_jitter,
                                legend, ax, **kwargs)
                
        elif type=='line':
            if kwargs.__len__() == 0:
                sns.lineplot(x, y, hue, size, style, data, palette, hue_order, hue_norm, sizes, size_order, size_norm, 
                             dashes, markers, style_order, units, estimator, ci, n_boot, sort, 
                             err_style, err_kws,legend, ax)
            else:
                sns.lineplot( x, y, hue, size, style, data, palette, hue_order, hue_norm, sizes, size_order, size_norm, 
                             dashes, markers, style_order, units, estimator, ci, n_boot, sort, 
                             err_style, err_kws, legend, ax, **kwargs)
                
                
               
                
                
                
                
    except Exception as e:
        print('Exception occurred in bivariate() in data quality module')
        print(e)


def multivariate(data,plt=None,sns=None, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None,
                 fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, 
                 cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None,
                 **kwargs):
    
    '''
    This is an Axes-level function and will draw the heatmap into the currently-active Axes if none is provided to the ax argument. Part of this Axes space will be taken and used to plot a colormap, unless cbar is False or a separate Axes is provided to cbar_ax.
    
    Parameters:	
        
    data : DataFrame, array, or list of arrays, optional    
        
    plt : matplotlib object containing specific properties, optional
    
    sns : seaborn object containing specific properties, optional
    
    vmin, vmax : floats, optional
    
    Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
    
    cmap : matplotlib colormap name or object, or list of colors, optional
    
    The mapping from data values to color space. If not provided, the default will depend on whether center is set.
    
    center : float, optional

    The value at which to center the colormap when plotting divergant data. Using this parameter will change the default cmap if none is specified.

    robust : bool, optional

    If True and vmin or vmax are absent, the colormap range is computed with robust quantiles instead of the extreme values.

    annot : bool or rectangular dataset, optional

    If True, write the data value in each cell. If an array-like with the same shape as data, then use this to annotate the heatmap instead of the raw data.
    
    fmt : string, optional

    String formatting code to use when adding annotations.

    annot_kws : dict of key, value mappings, optional

    Keyword arguments for ax.text when annot is True.

    linewidths : float, optional
    
    Width of the lines that will divide each cell.
    
    linecolor : color, optional
    
    Color of the lines that will divide each cell.
    
    cbar : boolean, optional
    
    Whether to draw a colorbar.
    
    cbar_kws : dict of key, value mappings, optional
    
    Keyword arguments for fig.colorbar.
    
    cbar_ax : matplotlib Axes, optional
    
    Axes in which to draw the colorbar, otherwise take space from the main Axes.
    
    square : boolean, optional
    
    If True, set the Axes aspect to “equal” so each cell will be square-shaped.
    
    xticklabels, yticklabels : “auto”, bool, list-like, or int, optional
    
    If True, plot the column names of the dataframe. If False, don’t plot the column names. If list-like, plot these alternate labels as the xticklabels. If an integer, use the column names but plot only every n label. If “auto”, try to densely plot non-overlapping labels.
    
    mask : boolean array or DataFrame, optional
    
    If passed, data will not be shown in cells where mask is True. Cells with missing values are automatically masked.
    
    ax : matplotlib Axes, optional
    
    Axes in which to draw the plot, otherwise use the currently-active Axes.
    
    kwargs : other keyword arguments
    
    All other keyword arguments are passed to ax.pcolormesh.
    
    Returns:	
    ax : matplotlib Axes
    
    Axes object with the heatmap.
    
    
    '''
    
    
    try:
        if plt==None:
            plt=global_plt
        if sns==None:
            sns=global_sns
        if kwargs.__len__() == 0:
            sns.heatmap(data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, linewidths,
                        linecolor, cbar, cbar_kws, cbar_ax, square, xticklabels,yticklabels, mask, ax)
        else:
            sns.heatmap(data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, linewidths,
                        linecolor, cbar, cbar_kws, cbar_ax, square, xticklabels,yticklabels, mask, ax,kwargs)
            
    except Exception as e:
        print('Exception occurred in multivariate() in data quality module')
        print(e)


def profile_data(data, target):
    
    '''
    Generate a profile report from a Dataset stored as a pandas `DataFrame`.
    
    Type inference: detect the types of columns in a dataframe.
    
    Essentials: type, unique values, missing values
    
    Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range
    
    Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
    
    Most frequent values
    
    Histogram
    
    Correlations highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
    
    Missing values matrix, count, heatmap and dendrogram of missing values
    
    Parameters:	
        
    data : DataFrame 
    
    target : Target variable in the dataset. This is of type string.
    
    '''
    
    #return data.profile_report(style={'full_width':True})
    return pandas_profiling.ProfileReport(data,target)    


def stats(data):
    
    
    '''
    Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
    
    Analyzes both numeric and object series, as well as DataFrame column sets of mixed data types. The output will vary depending on what is provided. Refer to the notes below for more detail.
    
    Parameters:	
        
    percentiles : list-like of numbers, optional   
        
    The percentiles to include in the output. All should fall between 0 and 1. The default is [.25, .5, .75], which returns the 25th, 50th, and 75th percentiles.
    
    include : ‘all’, list-like of dtypes or None (default), optional
    
    A white list of data types to include in the result. Ignored for Series. Here are the options:

   ‘all’ : All columns of the input will be included in the output.
   
    A list-like of dtypes : Limits the results to the provided data types. To limit the result to numeric types submit numpy.number. To limit it instead to object columns submit the numpy.object data type. Strings can also be used in the style of select_dtypes (e.g. df.describe(include=['O'])). To select pandas categorical columns, use 'category'
    
    None (default) : The result will include all numeric columns.

    exclude : list-like of dtypes or None (default), optional
   
    A black list of data types to omit from the result. Ignored for Series. Here are the options:

    A list-like of dtypes : Excludes the provided data types from the result. To exclude numeric types submit numpy.number. To exclude object columns submit the data type numpy.object. Strings can also be used in the style of select_dtypes (e.g. df.describe(include=['O'])). To exclude pandas categorical columns, use 'category'
    
    None (default) : The result will exclude nothing.
    
    Returns:	
    Series or DataFrame
    
    Summary statistics of the Series or Dataframe provided.
    
    
    '''
    return data.describe() 



def outlier_detection(df, method = 'dbscan', behaviour = 'new', max_samples= 1000, random_state=42, contamination_iso = 'auto',
                       eps = .6, metric='euclidean', min_samples = 5, n_jobs = -1, n_neighbors=30, contamination_lof = 'auto',  
                       contamination_ee = 0.1, nu=.001, kernel='rbf', gamma='auto'):
    
    '''
    This function helps to detect and remove the outliers. 
    
    Parameters:	
    df : Dataframe
    
    method : method to be chosen for outlier detection
    
    Uses iso_forest (Isolation Forest), dbscan (DBSCAN), lof (Local Outlier Factor), ee (Elliptic Envelope) and 
    svm (Support Vector Machine) Algorithms.
    
    Algorithm: Isolation Forest
    Parameters:	
    behaviour : str, default=’deprecated’
    
    This parameter has not effect, is deprecated, and will be removed.
    
    max_samples : int or float, optional (default=”auto”)
    
    The number of samples to draw from X to train each base estimator.
    
    If int, then draw max_samples samples.

    If float, then draw max_samples * X.shape[0] samples.
 
    If “auto”, then max_samples=min(256, n_samples).
    
    If max_samples is larger than the number of samples provided, all samples will be used for all trees (no sampling).
    
    random_state : int, RandomState instance or None, optional (default=None)
    
    If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the 
    random number generator; If None, the random number generator is the RandomState instance used by np.random.
    
    contamination_iso : ‘auto’ or float, optional (default=’auto’)
    
    The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when 
    fitting to define the threshold on the scores of the samples.

    If ‘auto’, the threshold is determined as in the original paper.

    If float, the contamination should be in the range [0, 0.5].
    
    Algorithm: DBSCAN
    Parameters:	
    eps : float, default=0.6
    
    This value is expected fron the user. If not mentioned, default value '0.6' will be chosen.
    
    The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
    This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter 
    to choose appropriately for your data set and distance function.
    
    metric : string, or callable, default=’euclidean’
    
    The metric to use when calculating distance between instances in a feature array. If metric is a string or callable,
    it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric parameter. If metric is 
    “precomputed”, X is assumed to be a distance matrix and must be square. X may be a Glossary, in which case only 
    “nonzero” elements may be considered neighbors for DBSCAN.
    
    min_samples : int, default=5
    
    The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
    This includes the point itself.
    
    n_jobs : int or None, default=None
    
    The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
    
    Algorithm: Local Outlier Factor
    Parameters:	
    n_neighbors : int, optional (default=20)
    
    Number of neighbors to use by default for kneighbors queries. If n_neighbors is larger than the number of samples provided, 
    all samples will be used.
    
    contamination_lof : ‘auto’ or float, optional (default=’auto’)
    
    The amount of contamination of the data set, i.e. the proportion of outliers in the data set. When fitting this is used 
    to define the threshold on the scores of the samples.

    if ‘auto’, the threshold is determined as in the original paper,

    if a float, the contamination should be in the range [0, 0.5].
    
    Algorithm: Elliptic Envelope
    Parameters:	
    contamination_ee : float in (0., 0.5), (default=0.1)
    
    This value is expected fron the user. If not mentioned, default value '0.1' will be chosen. 
    
    The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    
    Algorithm: Support Vector Machine
    Parameters:	
    nu : float 
    
    This value is expected fron the user. By default 0.5 will be taken.
    
    An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the 
    interval (0, 1].
    
    kernel : string, optional (default=’rbf’)
    
    Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
    or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.
    
    gamma : {‘scale’, ‘auto’} or float, optional (default=’scale’)
    
    Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

    if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,

    if ‘auto’, uses 1 / n_features.
    
    
    Returns:	
    df_without : Dataframe
    
    Returns the original dataframe without the outliers
    
    outliers : Dataframe
    
    Returns only the outliers
    
    
    ''' 
    
    scaler = MinMaxScaler() 
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    
    
    temp_df =df
    temp_df.dropna(inplace=True)
    
  
    try:
        if method == 'isoforest' :
            clf = IsolationForest(behaviour = 'new', max_samples=1000, random_state=random_state, contamination = contamination_iso)
            clf.fit(temp_df)
            x_pred_col = clf.predict(temp_df)
              
        elif method =='dbscan':
            outlier_detection = DBSCAN(eps = .6, metric='euclidean', min_samples = 5, n_jobs = -1)
            x_pred_col = outlier_detection.fit_predict(temp_df)
            x_pred_col = np.where(x_pred_col != -1, 1, x_pred_col) 
            
        elif method == 'lof' :
            lof = LocalOutlierFactor(n_neighbors=30, contamination = contamination_lof)
            x_pred_col = lof.fit_predict(temp_df) 
        
        elif method == 'ee' :
            ee = EllipticEnvelope(contamination = contamination_ee,random_state=random_state)
            ee.fit(temp_df) 
            x_pred_col = ee.predict(temp_df)
            
        elif method == 'svm' :
            srbf =svm.OneClassSVM(nu=.001,kernel='rbf',gamma='auto')
            srbf.fit(temp_df)
            x_pred_col =srbf.predict(temp_df)
                   
                
        new_col='outlier prediction'  # temporary column to mark Outliers    
        #temp_df[new_col]=temp_df[x_col_o]
        temp_df[new_col]=x_pred_col   # move predicted outliers to temporary dataframe
        
        outliers=temp_df.loc[temp_df[new_col]==-1]   # Move outliers to new dataframe
        outliers.drop([new_col],axis=1, inplace = True)
        outliers_ind=list(outliers.index)            # indices of the all outliers
    
    
        df_without = pd.merge(df,outliers, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        df_without.drop([new_col],axis=1, inplace = True)
        outliers_lab = list(x_pred_col)    
     

    except Exception as e:
        print('Exception occurred in outlier_detection() in data quality module')
        print(e)
    
    return df_without, outliers, outliers_lab




def impute(entire_data,col_name=None, col_type='categorical',algo_name = 'RFC'): 
    try:
        if(col_name==None):
            col_list = entire_data.columns[entire_data.isna().any()].tolist()
            #col_list = entire_data.columns
            col_types = [entire_data[col].dtype for col in col_list]
            list_indices = range(0,len(col_list))
            for val in list_indices:
                #print(0.1)
                #print(col_list[val])
                #print(col_types[val])
                if (col_types[val] not in ['int64','float64']):
                    #print(0.2)
                    #print(col_list[val])
                    #print(col_types[val])
                    entire_data[col_list[val]] = entire_data[col_list[val]].astype('float64')
                    entire_data = impute_columnwise(entire_data,col_list[val],col_type,algo_name)
                else:
                    #print(0.3)
                    #print(col_list[val])
                    #print(col_types[val])
                    entire_data = impute_columnwise(entire_data,col_list[val],col_type='continuous',algo_name = 'LR')
                    
        else:
            #print(0.4)
            entire_data = impute_columnwise(entire_data,col_name,col_type,algo_name)
        return entire_data
    except Exception as e:
        print('Exception occurred in impute() in data quality module')
        print(e)







def impute_columnwise(entire_data,col_name, col_type='categorical',algo_name = 'RFC'):    
    
    try:
       
        #print('1')
        #print(entire_data.columns)
        #print(entire_data.shape)
        
        
        duplicates_removed_data = remove_duplicates(entire_data)
       
        
        
        #print('2')
        #print(entire_data.shape)
        
        #print(duplicates_removed_data.columns)
        #print(duplicates_removed_data.shape)
        new_data = duplicates_removed_data
        
        if(len(duplicates_removed_data) > 1000):
            #print('2.1')
            #print(entire_data.shape)
            
            #entire_data1 = entire_data
            temp_data = duplicates_removed_data.dropna(axis=0)
            #print('2.1.0')
            #print(temp_data.head(10))
            new_data = remove_multicollinearity(temp_data, col_name)
            #new_data = remove_multicollinearity(duplicates_removed_data.dropna(),categ_to_numeric(duplicates_removed_data.dropna().drop(col_name)) col_name)
            
            
            
            
            
            #print('inside impute method, after calling remove_multicollinearity()')
            #print(new_data.columns)
            #print('2.1.1')
            #print(entire_data.shape)
            new_data = imp_features(col_name, duplicates_removed_data[new_data.columns])
            #print('2.1.2')
            #print(entire_data1.shape)
            
            #new_data = imp_features(col_name, duplicates_removed_data)
            
            
            
            
            #print(new_data.columns)
            #print(new_data.shape)
            #print('2.2')
            #print(entire_data1.shape)
            #print(new_data.columns)
            #print(new_data.shape)
            new_data = random_records(new_data)
            #print('2.3')
            #print(entire_data1.shape)
        #print(new_data.columns)
        #print(new_data.shape)
        #print('3')
        #print(entire_data.shape)
        # split 
        #from sklearn.model_selection import train_test_split
        X = new_data.drop(col_name, axis =1)  
        y = new_data[col_name]
        #print(X.columns)
        #print(X.shape)
        #print(y.shape)
        #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
        numeric_data = categ_to_numeric(X)
        #print('4')
        #print(numeric_data.columns)
        #print(numeric_data.shape)
        if(col_type=='categorical' and algo_name == 'RFC'):
            #print('5')
            return classifier_impute(col_name, numeric_data,y, entire_data, algo_name = 'RFC')
        else:
            #print('6')
            #print(entire_data1.shape)
            return regression_impute(col_name, numeric_data,y, entire_data, algo_name = 'LR')
    
    except Exception as e:
        print('Exception occurred in impute_columnwise() in data quality module')
        print(e)
        
        
def imp_features(col_name, entire_data):
    #print('4.1.1')
    #print(entire_data.columns)
    #print(entire_data.shape)
    data = entire_data.dropna()
    if(data[col_name].dtype not in ['int64','float64']):
        data[col_name] = data[col_name].astype('float64')
    X = data.drop(col_name, axis=1)  
    y = data[col_name]
    X = initial_impute(X)
    #print('4.1.2')
    #print(X.columns)
    #print(X.shape)
    rfc = RandomForestClassifier(random_state = 0, n_jobs = -1)
    #print('4.1.3')
    rfc.fit(X, y)
    #print('4.1.4')
    df_imp = pd.DataFrame({'Column_name':X.columns, 'imp':rfc.feature_importances_}) 
    #print('4.1.5')
    #print(rfc.feature_importances_)
    #print(type(df_imp))
    #print(df_imp)
    #print(df_imp.columns)
    #print(df_imp.shape)
    #df_imp = df_imp.sort_values('imp', ascending=False)
    #print('After sorting')
    #print(df_imp)
    df_imp_ts = df_imp.nlargest(int((0.6)*(df_imp.shape[0])),'imp') 
    #print('2.1.4')
    #print(type(df_imp_ts))
    #print(df_imp_ts.columns)
    #print(df_imp_ts.shape)
    df_imp_ts = entire_data[df_imp_ts['Column_name']]
    #print(df_imp_ts)
    df_imp_ts[col_name] = y
    #print(df_imp_ts.shape)
    return df_imp_ts

def random_records(data):
    #print('inside random records 1')
    rndm_data = data.sample(frac=0.7)
    #print('inside random records 2')
    return rndm_data

def remove_duplicates(data):
    data.drop_duplicates(subset = None,keep = 'first', inplace = True) 
    return data


def categ_to_numeric(data):
    #print('3.1')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    return data


def classifier_impute(col_name, numeric_data, y, entire_data, algo_name = 'RFC'):
    #print('5.1')
    numeric_data = initial_impute(numeric_data)
    numeric_data[col_name] = y
    #print(numeric_data.columns)
    #print('5.2')
    #print(numeric_data.shape)
    train = numeric_data.dropna()
    #print('5.3')
    X_train = train.drop(col_name, axis =1)
    
    y_train = train[col_name]
    #print('5.4')
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    #print('5.5')
    
    entire_data_imp_features = entire_data[numeric_data.columns]
    
    test = entire_data_imp_features[entire_data_imp_features[col_name].isnull()]
    
    X = test.drop(col_name, axis =1)  
    y = test[col_name]    
    X= initial_impute(X)
    X= categ_to_numeric(X)
    
    #print(test.shape)
    #print(test)
    #print('5.6')
    if(X.shape[0]!=0):
        #print('5.5.1')
        for i in list(X.index):
            #print('5.5.2')
            #print(i)
            temp_data_list = X.loc[i]
            #print(temp_data_list)
            #print(type(temp_data_list))
            #temp_data_list.drop(col_name,inplace=True)
            keys = temp_data_list.keys()
            #print(keys)
            #print(temp_data_list.values)
            if(type(entire_data[col_name])=='int64'):
                entire_data.loc[i,col_name]=int(model.predict((temp_data_list.values).reshape(1, -1)))
            else:
                entire_data.loc[i,col_name]=model.predict((temp_data_list.values).reshape(1, -1))
        #X_test = test.drop(col_name, axis =1)
        #y_test = model.predict(X_test)
        #print(entire_data.head())
        return entire_data
    else:
        print('There was no data for imputing')
        return entire_data

def regression_impute(col_name, numeric_data,y, entire_data, algo_name = 'LR'):
    
    #print('6.1')
    numeric_data = initial_impute(numeric_data)
    numeric_data[col_name] = y
    #print(numeric_data.columns)
    #print('6.2')
    #print(numeric_data.shape)
    #print('6.2.1')
    #print(entire_data.shape)
    train = numeric_data.dropna()
    #print('6.3')
    X_train = train.drop(col_name, axis =1)
    
    y_train = train[col_name]
    #print('6.4')
    model = linear_model.LinearRegression()
    model.fit(X_train,y_train)
    #print('6.5')
    #print(entire_data.shape)
    entire_data_imp_features = entire_data[numeric_data.columns]
    #print('6.5.1')
    #print(entire_data_imp_features.shape)
    test = entire_data_imp_features[entire_data_imp_features[col_name].isnull()]
    #print('Inside regression_impute')
    #print(test.shape)
    
    X = test.drop(col_name, axis =1)  
    y = test[col_name]    
    X= initial_impute(X)
    X= categ_to_numeric(X)
    
    #print(test.shape)
    #print(test)
    #print('6.6')
    if(X.shape[0]!=0):
        #print('6.6.1')
        for i in list(X.index):
            #print('6.6.2')
            #print(i)
            temp_data_list = X.loc[i]
            #print(temp_data_list)
            #print(type(temp_data_list))
            #temp_data_list.drop(col_name,inplace=True)
            keys = temp_data_list.keys()
            #print(keys)
            #print(temp_data_list.values)
            if(entire_data[col_name].dtype=='float64' ):
                entire_data.loc[i,col_name]=model.predict((temp_data_list.values).reshape(1, -1)).astype('int')
                #entire_data[col_name] = entire_data[col_name].astype('int')
            else:
                entire_data.loc[i,col_name]=model.predict((temp_data_list.values).reshape(1, -1))
        #X_test = test.drop(col_name, axis =1)
        #y_test = model.predict(X_test)
        #print(entire_data.head())
        #print('6.6.3')
        return entire_data
    else:
        print('There was no data for imputing')
        return entire_data


def noise_detection(df, method = 'savgol', window_length = 101, polyorder = 2, n_lf = 15) :
    
    '''
    This method helps to detect and reduce the noise in the data. 
    
    This method is suited for removing the noise in time series related datasets.
    
    Parameters:	
    df : Dataframe
    
    method : method to be chosen for noise reduction 
    
    Uses savgol (Savitzky-Golay) and lfilter (Infinite impulse response or Finite impulse response) filtering techniques.
    
    Technique: Savitzky-Golay
    Parameters:	
    window_length : int
    
    The length of the filter window (i.e. the number of coefficients). window_length must be a positive odd integer.
    
    This value is expected fron the user. If not mentioned, default value '101' will be chosen.
    
    polyorder : int
    
    The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    
    This value is expected fron the user. If not mentioned, default value '2' will be chosen.
    
    Technique: Infinite impulse response or Finite impulse response
    Parameters:	
    n_lf : int
    
    The value on which the numerator coefficient vector in a 1-D sequence is dependent on. Larger the value of n_lf, smoother
    the curve.
    
    This value is expected fron the user. If not mentioned, default value '100' will be chosen.

    
    Returns:	
    df : Dataframe  
    Returns the dataframe after reducing the noise.
    
    
    '''

    df.dropna(inplace=True)
    
    
    try:
        for feature in df.columns:
            if method == 'savgol' :
                if (df[feature].dtype == 'int32' or 'int64') :
                    df[feature] = (savgol_filter(df[feature], window_length = window_length, polyorder = polyorder)).astype(df[feature].dtype)
                elif (df[feature].dtype == 'float32' or 'float64'):
                    df[feature] = (savgol_filter(df[feature], window_length = window_length, polyorder = polyorder)).astype(df[feature].dtype)
                
            
            elif method =='lfilter':
                n = n_lf  # the larger n is, the smoother curve will be
                b = [1.0 / n] * n
                a = 1
                if (df[feature].dtype == 'int32' or 'int64') :
                    df[feature] = (lfilter(b,a,df[feature])).astype(df[feature].dtype)
                elif (df[feature].dtype == 'float32' or 'float64') :
                    df[feature] = (lfilter(b,a,df[feature])).astype(df[feature].dtype)
                
    except Exception as e:
        print('Exception occurred in noise_detection() in data quality module')
        print(e)
    
    return df
    

def initial_impute(convrtd_data):
    return convrtd_data.fillna(convrtd_data.mean())



def remove_multicollinearity(dataset, target, threshold = 10.0):
    
    '''
    This method helps to detect multicollinearity and remove it. 
    
    Parameters:	
    dataset : Dataframe
    
    target: Dataframe
    Target variable in the dataset. Should be mentioned in the form of string.
    
    threshold: int/float
    The threshold value for Variance Inflation Factor
    This parameter is expected from the user. By default, '10.0' is taken.
    
    Returns:	
    X : Dataframe  
    Returns the dataframe after removing the multicollinearity.
    
    '''
    
    #dataset.dropna(inplace = True)
    X = dataset.drop([target], axis = 1)
    y = dataset[target]
    
    try:
        dropped=True
        while dropped:
            variables = X.columns
            non_numeric_data_cols = [col for col in X.columns if X[col].dtype not in ['int64','float64'] ]
            #print('M')
            #print(non_numeric_data_cols)
            for col in non_numeric_data_cols:
                #print('M.1')
                X[col] = X[col].astype('float64')
                #print(X[col])
                #print(X[col].dtype)
            dropped = False
            #print('M.2')
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            #print('M.3')
            max_vif = max(vif)
            if max_vif > threshold:
                #print('M.4')
                maxloc = vif.index(max_vif)
                #print('M.5')
                #print(f'Dropping {X.columns[maxloc]} with vif= {max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                #print('M.6')
                dropped=True
            #print('M.7')
           
        X[target] = y
    except Exception as e:
        print('Exception occurred in remove_multicollinearity() in data quality module')
        print(e)
    return X


#def skewness(data_column):
#    return skew(data_column)
    
#from scipy.stats import skew, kurtosis


def skewness(data_column,data):
    data.dropna(inplace = True)
    
    try:
        s = skew(data[data_column])
        #print ('Skewness of '+data_column+' is = ',s)
        mean = np.mean(data[data_column])
        median = np.median(data[data_column])
        if mean > median:
            print('The '+data_column+' feature is positively skewed.')
            print('A positively skewed distribution has a long right tail.')
            
        elif mean < median:
            print('The '+data_column+' feature is negatively skewed.')
            print('A negatively skewed distribution has a long left tail.')
        return s
        
    except Exception as e:
        print('Exception occurred in skewness() in data quality module')
        print(e)
        
    








#def find_kurtosis(data_column):
#    return kurtosis(data_column)


def find_kurtosis(data_column,data):
    data.dropna(inplace = True)
    try:
        k = kurtosis(data[data_column])
        #print ('Kurtosis of '+data_column+' is = ',k)
       
        if k == 0:
            print('The '+data_column+' feature distribution pattern is Mesokurtic.')
            print('Mesokurtic distributions means that the data follows a normal distribution.')
           
        elif k < 0:
            print('The '+data_column+' feature distribution pattern is Platykurtic.')
            print('Platykurtic distributions are flatter than a normal distribution with shorter tails.')
           
        elif k > 0:
            print('The '+data_column+' feature distribution pattern is Leptokurtic.')
            print('Leptokurtic distributions are more peaked than a normal distribution with longer tails.')
        return k
       
    except Exception as e:
        print('Exception occurred in kurtosis() in data quality module')
        print(e)
       
    

def balance_data(df, target, method = 'over_sampling'):
    
    
    '''
    This function helps to balance the imbalanced dataset. 
    
    Parameters:	
    df : Dataframe
    
    target : Dependent variable or target variable
    This parameter should be passed as a string.
    
    method : method to be chosen for balancing the dataset
    
    Uses over_sampling (SMOTE) and under_sampling (NearMiss) sampling techniques.
    
    Returns:	
    df_new : Dataframe  
    Returns the new dataframe after balancing.
    
    
    '''
    
    
    df.dropna(inplace = True)
    X = df.drop([target], axis = 1)
    y = df[target]
    print("Distribution of 1's in target variable before balancing = ", sum(y == 1))
    print("Distribution of 0's in target variable before balancing = ", sum(y == 0))
    
    try:
        if method == 'over_sampling':
            smt = SMOTE()
            X1, y1 = smt.fit_sample(X, y)
            print("Distribution of 1's in target variable after over sampling = ", sum(y1 == 1))
            print("Distribution of 0's in target variable after over sampling = ", sum(y1 == 0))
            df_X1 = pd.DataFrame(X1, columns = X.columns)
            df_X1[target] = y1
            df_new = df_X1
            return df_new
            
        
        elif method == 'under_sampling':
            nmiss = NearMiss()
            X1, y1 = nmiss.fit_sample(X, y)
            print("Distribution of 1's in target variable after under sampling = ",sum(y1 == 1))
            print("Distribution of 0's in target variable after under sampling = ",sum(y1 == 0))
            df_X1 = pd.DataFrame(X1, columns = X.columns)
            df_X1[target] = y1
            df_new = df_X1
            return df_new
    
    except Exception as e:
        print('Exception occurred in data_balancing() in data quality module')
        print(e)
        
def correlations(data, method = 'pearson'):
    try:
        if method == 'pearson':
            cor_mat = data.corr(method = 'pearson')
        elif method == 'spearman':
            cor_mat = data.corr(method = 'spearman')
        elif method == 'kendall':
            cor_mat = data.corr(method = 'kendall')
        f,ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(cor_mat, cbar = True,  square = True, annot = True, fmt= '.1f', xticklabels= True, yticklabels= True, 
                    cmap="coolwarm", linewidths=.5, ax=ax)
        plt.title('CORRELATION MATRIX - HEATMAP', size=20)
        return cor_mat
        
    except Exception as e:
        print('Exception occurred in correlation_matrix() in data quality module')
        print(e)
    
def trends(column_name, data, ind_col, period=100):
    
    try:
        data_temp = data
        data_temp[ind_col] = pd.to_datetime(data_temp[ind_col])
        data_temp.set_index(data_temp[ind_col],inplace=True)
        data_ts = data_temp[[column_name]]
        ts = data_ts
        rcParams['figure.figsize'] = (10,6)
        decomposition = sm.tsa.seasonal_decompose(ts, period=period)
        fig = decomposition.plot()
        dftest = adfuller(ts)
        print('ADF test statistic for'+' '+column_name+' is = ', dftest[0])
        print('p-value for ADF test for'+' '+column_name+' is = ', dftest[1])
        if dftest[1] > 0.05:
            print('As the p-value is greater than 0.05, we fail to reject the null hypothesis (H0), the '+column_name+' feature has a unit root and is non-stationary.')
        elif dftest[1] < 0.05:
            print('As the p-value is less than 0.05, we reject the null hypothesis (H0), the '+column_name+ ' feature has no unit root and is stationary.')
        print('Figure below shows the decomposed Trend, Seasonality and Residual patterns along with the observed pattern for '+column_name+ ' feature.')
        
        return fig.show()
        
    except Exception as e:
        print('Exception occurred in trend_identification() in data quality module')
        print(e)
        
    



#df = pd.read_csv('CustomerChurnDataProfiled.csv')
#df = pd.read_csv('UCI_Credit_Card.csv')



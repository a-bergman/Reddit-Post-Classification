# Imports 

import pandas            as pd
import seaborn           as sns
import numpy             as np
import matplotlib.pyplot as plt
import plotly.io         as pio
import plotly.graph_objs as go
from sklearn.metrics     import roc_auc_score

# Setting the basic appearance for the graphs

# For Seaborn
sns.set(style = "white", palette = "deep")

# For Plotly
pio.templates.default = "simple_white"

"""

The docstrings for each graph contain the following:

- parameters  : values which must be entered, some of which have defaults
- description : what each function does
- returns     : the output of each function

The parameters section of each docstring is set up as:

parameter : definition : type : possible values (if applicable)

Each function is designed to output n number of graphs where n > 1, but can output a single graph.
The only function which is not designed for multiple outputs is the KDE function which only outputs a single graph of two columns.

"""
# Numeric Graphs

def histograms(df, columns, titles, labels, ylabel, ticks, dim, row, col):
    """
    Parameters:
    -----------
    df      : the dataframe source of data               : dataframe :       
    columns : list of columns to be plotted              : str       :
    titles  : list of titles for each plot               : str       :
    labels  : list of x-labels for each plot             : str       :
    label   : the y-label for each plot                  : str       :
    ticks   : the list of ranges for each plot's x-ticks : np.arange :
    dim     : tuple of the dimensions of each plot       : int       :
    row     : how many rows will be generated            : int       :
    col     : how many columns will be generated         : int       :

    Description:
    ------------
    Plots histograms for columns containing continuous data in a Pandas dataframe and gives the user greater customization for each plot.

    Returns:
    --------
    Creates n number of histograms arranged by the input rows and columns.
    """
    count = 0
    fig = plt.figure(figsize = dim, facecolor = "white")
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        plt.title(f"Distribution Of {titles[c]}", size = 18)
        sns.distplot(df[column], color = "black", kde = False)
        plt.axvline(df[column].mean(), color = "red")
        plt.xlabel(f"{labels[c]}", size = 16)
        plt.ylabel(f"{ylabel}", size = 16)
        plt.xticks(ticks = ticks[c], size = 14)
        plt.yticks(size = 14)
    plt.tight_layout()
    plt.show();

def histogram_2d(x, y, df, title, xlabel, ylabel, xticks, yticks, ytitle = 0.9, width = 1000, height = 550, colorscale = "RdBu"):
    """
    Parameters:
    -----------
    x          : the column to be plotted on the x-axis        : str       : :
    y          : the column to be plotted on the y-axis        : str       : :
    df         : dataframe source of data                      : dataframe : :
    title      : title for the graph                           : str       : :
    xlabel     : label for the x-axis                          : str       : :
    ylabel     : label for the y-axis                          : str       : :
    xticks     : tick range for the x-axis                     : np.arange : :
    yticks     : tick range for the y-axis                     : np.arange : :
    ytitle     : how high above the plot to place the title    : float     : :
    width      : width of the plot                             : int       : :
    height     : height of the plot                            : int       : :
    colorscale : color scale to map overlapping of data points : str       : :

    Description:
    ------------
    Plots two continuous columns against each other and shows overlapping points with a color scale and gives the user greater customization over the plot.

    Returns:
    --------
    Creates a single two-dimensional histogram with the input dimensions.
    """
    fig = go.Figure(go.Histogram2d(x = df[x],y = df[y],colorscale = colorscale))
    fig.update_layout(title = dict(text = title, y = ytitle, x = 0.5, xanchor = "center", yanchor = "top"),
                      xaxis_title = xlabel, yaxis_title = ylabel, width = width, height = height)
    fig.update_xaxes(tickvals = xticks, title_font = dict(size = 18),tickfont = dict(size = 14))
    fig.update_yaxes(tickvals = yticks, title_font = dict(size = 18),tickfont = dict(size = 14))
    fig.show()

def kdeplots(df, cols, title, dim, colors, labels, xlabel, ylabel, ticks, shade = True):
    """
    Parameters:
    -----------
    df     : the dataframe source of data                   : dataframe :
    cols   : list of the columns                            : str       :
    title  : plot title                                     : str       :
    dim    : tuple of the dimensions of each plot           : int       :
    colors : list of the colors of each kde plot            : str       :
    labels : list of thethe name of each kde                : str       :
    xlabel : the label of the x-axis                        : str       : 
    ylabel : the label of the y-axis                        : str       :
    ticks  : the range of the x-ticks                       : np.arange :
    shade  : whether or not to shade the area under the kde : Bool      :

    Description:
    ------------
    Overlays two univariate kernel density estimates on the same axis which estimate the distribution of two columns of data.

    Returns:
    --------
    A single graph with two overlaid density estimates.
    
    """
    plt.figure(figsize = dim, facecolor = "white")
    plt.title(f"{title}", size = 18)
    sns.kdeplot(df[cols[0]], shade = shade, color = colors[0], label = labels[0])
    sns.kdeplot(df[cols[1]], shade = shade, color = colors[1], label = labels[1])
    plt.xlabel(f"{xlabel}", size = 16)
    plt.ylabel(f"{ylabel}", size = 16)
    plt.xticks(ticks, size = 14)
    plt.yticks(size = 14)
    plt.legend(bbox_to_anchor = (1.04, 1), loc = "upper left", fontsize = 16);

def boxplots(df, columns, titles, labels, ticks, dim, row, col, orient = "h"):
    """
    Parameters:
    -----------
    df      : dataframe source of the data         : dataframe :
    columns : list of the columns to be plotted    : str       :
    titles  : list of titles for each plot         : str       :
    ticks   : list of ranges for the x-ticks       : np.arange :
    dim     : tuple of the dimensions of each plot : int       :
    row     : how many rows will be generated      : int       :
    col     : how many columns will be generated   : int       :
    orient  : orientation of each plot             : str       : "h"|"v"

    Description:
    ------------
    Plots boxplots for columns containing continuous data in a Pandas dataframe and gives the user greater customization for each plot.

    Returns:
    --------
    n number of boxplots arranged by the input rows and columns.
    """
    count = 0
    fig = plt.figure(figsize = dim, facecolor = "white")
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        plt.title(f"{titles[c]}", size = 18)
        sns.boxplot(x = column, data = df, orient = orient)
        plt.xlabel(f"{labels[c]}", size = 16)
        plt.xticks(ticks = ticks[c], size = 14)
        plt.yticks(size = 14)
    plt.tight_layout()
    plt.show();

def violinplots(df, columns, titles, labels, ticks, dim, row, col, orient = "h"):
    """
    Parameters:
    -----------
    df      : dataframe source of data             : dataframe :
    columns : list of columns to be plotted        : str       :
    titles  : list of titles for each plot         : str       :
    labels  : list of the x-labels for each plot   : str       :
    ticks   : list of ranges for the x-ticks       : np.range  :
    dim     : tuple of the dimensions of each plot : int       :
    row     : how many rows will be generated      : int       :
    col     : how many columns will be generated   : int       :
    orient  : orientation of each plot             : str       : "h"|"v"

    Descriptions:
    -------------
    Plots violin plots for columns containing data in a Pandas dataframe and gives the user greater customization for each plot.
    An improvement over the standard box plot in that it plots a kernel density plot of the points on the sides of each plot.

    Returns:
    --------
    n number of violin plots arranged by the input rows and columns.
    """
    count = 0
    fig = plt.figure(figsize = dim, facecolor = "white")
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        plt.title(f"{titles[c]}", size = 18)
        sns.violinplot(x = column, data = df, orient = orient)
        plt.xlabel(f"{labels[c]}", size = 16)
        plt.xticks(ticks = ticks[c], size = 14)
        plt.yticks(size = 14)
    plt.tight_layout()
    plt.show();

def regressionplots(df, columns, y, titles, labels, ylabel, ticks, dim, row, col, mark = "*", color = "black", kws = {"color": "red"}, ci = None):
    """
    Parameters:
    -----------
    df      : dataframe source of data                         : dataframe :
    columns : the list of columns to be plotted                : str       :
    y       : the column against which the columns are plotted : str       :
    titles  : list of the titles for each plot                 : str       :
    ylabel  : the title of the y-axis                          : str       :
    ticks   : list of ranges of x-ticks for each plot          : np.arange :
    dim     : tuple of the dimensions of each plot             : int       :
    row     : how many rows will be generated                  : int       :
    col     : how many columns will be generated               : int       : 
    mark    : what character the markers will be               : str       :
    color   : what color the markers are                       : str       :
    kws     : what color the regression line is                : dict      :
    ci      : whether or not to plot a confidence interval     : Bool      :

    Description:
    ------------
    Plots a scatter plot for each column of continuous data in a Pandas dataframe with a regression line 
    and allows the user to have greater control of the appearance of each graph.

    Returns:
    --------
    n number of regression plots arranged by the input rows and columns.
    """
    count = 0
    fig = plt.figure(figsize = dim, facecolor = "white")
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        plt.title(f"{titles[c]}", size = 18)
        sns.regplot(x = column, y = y, data = df, fit_reg = True,  marker = mark, color = color, line_kws = kws, ci = ci)
        plt.xlabel(f"{labels[c]}", size = 16)
        plt.ylabel(f"{ylabel}", size = 16)
        plt.xticks(ticks = ticks[c], size = 14)
        plt.yticks(size = 14)
    plt.tight_layout()
    plt.show();

def heatmap(df, columns, dim, title, vmin, vmax, cmap = "RdBu", annot = True):
    """
    Parameters:
    -----------
    df      : dataframe source of the data                  : dataframe :
    columns : list of the columns to be included            : str       :
    dim     : tuple of the dimensions of the graph          : int       :
    title   : title of the graph                            : str       :
    vmin    : minimum correlation value                     : int       :
    vmax    : maximum correlation value                     : int       :
    cmap    : the color scheme to be used                   : str       :
    annot   : whether or not the heat map will be annotated : Bool      :
    
    Description:
    ------------
    Plots a heatmap for columns containing continuous data in a Pandas dataframe and allows for increased appearance control.
    The resulting heatmap is not mirrored

    Returns:
    --------
    A heat map displaying the correlations between n number of columns.
    """
    plt.figure(figsize = dim, facecolor = "white")
    plt.title(f"{title}", size = 18)
    corr = df[columns].corr()
    mask = np.zeros_like(corr)                                                                                
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(corr, cmap = cmap,  mask = mask, vmin = vmin, vmax = vmax, annot = annot)
    plt.xticks(size = 14)
    plt.yticks(size = 14);

# Categorical Graphs

def countplots(df, columns, titles, labels, ylabel, dim, row, col, orient = "h", hue = None):
    """
    Parameters:
    -----------
    df      : dataframe source of data                    : dataframe :
    columns : list of the columns to be plotted           : str       :
    titles  : list of the titles for each plot            : str       :
    labels  : list of the x-labels for each plot          : str       :
    ylabel  : list of the ylabel for each plt             : str       :
    dim     : tuple of the dimensions of each plot        : int       :
    row     : how many rows will be generated             : int       :
    col     : how many columns will be generated          : int       :
    ci      : whether or not to add a confidence interval : Bool/str  : "sd"
    orient  : orientation of each plot                    : str       : "h"|"v"
    hue     : which column will be used for color-coding  : str       : 
    
    Description:
    -------------   
    Creates a count plot for columns in a Pandas dataframe containing categorical data.  
    This type of plot explicitly counts the categories in a dataframe column.

    Returns:
    --------
    n number of count plots arranged by the input rows and columns.
    """
    fig = plt.figure(figsize = dim, facecolor = "white")
    count = 0
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        title = titles[c]
        plt.title(f"{title}", size = 18)
        sns.countplot(x = column, data = df, orient = orient, hue = hue)
        plt.xlabel(f"{labels[c]}", size = 16)
        plt.ylabel(f"{ylabel}", size = 16)
        plt.xticks(size = 14)
        plt.yticks(size = 14)
    plt.tight_layout()
    plt.show();

def barplots(df, columns, y, labels, ylabel, titles, dim, row, col, ci = False, orient = "v", hue = None):
    """
    Parameters:
    -----------
    df     : dataframe source of data                   : dataframe :
    x      : list of the x inputs for each plot         : str       :
    y      : list of the y input for each plot          : str       :
    labels : list of the x-labels for each plot         : str       :
    ylabel : y-label for each plot                      : str       :
    titles : list of the titles for each plot           : strs      :
    dim    : tuple of the dimensions of each plot       : int       :
    row    : how many rows will be generated            : int       :
    col    : how many columns will be generated         : int       :
    ci     : whether or not to add confidence interval  : Bool      : "sd"
    orient : orientation of each bar plot               : str       : "h"|"v"
    hue    : which column will be used for color-coding : str       :

    Description:
    ------------
    Plots a bar plot for each column containing categorical data in a Pandas dataframe and allows for greater appearance control.
    This type of plot takes a categorical variable and returns the mean of a corresponding numeric variable.

    Returns:
    n number of barplots arranged by the input rows and columns.
    """
    fig = plt.figure(figsize = dim, facecolor = "white")
    count = 0
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        title = titles[c]
        plt.title(f"{title}", size = 18)
        sns.barplot(x = column, y = y, data = df, ci = ci, orient = orient, hue = hue)
        plt.xlabel(f"{labels[c]}", size = 16)
        plt.ylabel(f"{ylabel}", size = 16)
        plt.xticks(size = 14)
        plt.yticks(size = 14)
    plt.tight_layout()
    plt.show();

def barplot(df, x, y, title, label, ylabel, dim, orient = "v", ci = False, hue = None):
    """
    Parameters:
    -----------
    df     : dataframe source of the data               : dataframe :
    x      : the column to be the x-axis                : str       :
    y      : the column to be the y-axis                : str       :
    title  : title of the graph                         : str       :
    label  : the label of the x-axis                    : str       :
    ylabel : the label of the y-axis                    : str       :
    dim    : tuple of the graph dimensions              : int       :
    orient : orientation of the graph                   : str       : "h"|"v"
    ci     : whether or not to add confidence interval  : Bool      : "sd"
    hue    : which column will be used for color-coding : str       :  

    Description:
    ------------
    Plots a single bar chart for a categorical column in a Pandas dataframe and allows for greater appearance control.
    This type of chart takes a categorical variable and returns the mean of a corresponding numeric variable.

    Returns:
    --------
    A single bar plot with the input dimensions.
    """
    plt.figure(figsize = dim, facecolor = "white")
    sns.barplot(x = x, y = y, data = df, orient = orient, ci = ci, hue = hue)
    plt.title(f"{title}", size = 18)
    plt.xlabel(f"{label}", size = 16)
    plt.ylabel(f"{ylabel}", size = 16)
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    plt.tight_layout();

def categorical_boxplots(df, x, columns, titles, labels, ylabels, ticks, dim, row, col, hue = None, orient = "h"):
    """
    Parameters:
    -----------
    df      : dataframe source of the data         : dataframe :
    x       : categorical column to be the x-axis  : str       :
    columns : list of continuous columns           : str       :
    hue     : column to be used for color-coding   : str       :
    title   : list the titles for each plot        : str       :
    labels  : list of the xlabels for each plot    : str       :
    ylabels : list of the ylabels for each plot    : str       :
    ticks   : list of ranges for the y-ticks       : np.arange :
    dim     : tuple of the dimensions of each plot : int       :
    row     : how many rows will be generated      : int       :
    col     : how many columns will be generated   : int       :
    orient  : orientation of each plot             : str       : "h"|"v"

    Description:
    ------------
    Plots boxplots for columns containing continuous data in a Pandas dataframe and gives the user greater customization for each plot.

    Returns:
    --------
    n number of categorical boxplots arranged by the input rows and columns.
    """
    count = 0
    fig = plt.figure(figsize = dim, facecolor = "white")
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        plt.title(f"{titles[c]}", size = 18)
        sns.boxplot(x = x, y = column, data = df, hue = hue, orient = orient)
        plt.xlabel(f"{labels[c]}", size = 16)
        plt.ylabel(f"{ylabels[c]}", size = 16)
        plt.xticks(size = 14)
        plt.yticks(ticks = ticks[c], size = 14)
    plt.tight_layout()
    plt.show();

def categorical_violinplots(df, x, columns, hue, titles, labels, ylabels, ticks, dim, row, col, orient = "h", split = False):
    """
    Parameters:
    -----------
    df      : dataframe source of data                     : dataframe :
    x       : categorical column to be the x-axis          : str       :
    columns : list of numeric columns                      : str       :
    hue     : column to be used for color-coding           : str       :
    titles  : list of the titles for each plot             : str       :
    labels  : list of the xlabels for each plot            : str       :
    ylabels : list of the ylabels for each plot            : str       :
    ticks   : list of ranges for the x-ticks               : np.range  :
    dim     : tuple of the dimensions of each plot         : int       :
    row     : how many rows will be generated              : int       :
    col     : how many columns will be generated           : int       :
    orient  : orientation of each plot                     : str       : "h"|"v"
    split   : whether to not to spit each plot for the hue : Bool      :

    Descriptions:
    -------------
    Plots violin plots for columns containing data in a Pandas dataframe and gives the user greater customization for each plot.
    An improvement over the standard box plot in that it plots a kernel density plot of the points on the sides of each plot.

    Returns:
    --------
    n number of violin plots arranged by the input rows and columns.
    """
    count = 0
    fig = plt.figure(figsize = dim, facecolor = "white")
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        plt.title(f"{titles[c]}", size = 18)
        sns.violinplot(x = x, y = column, df = df, hue = hue, orient = orient, split = split)
        plt.xlabel(f"{labels[c]}", size = 16)
        plt.ylabel(f"{ylabels[c]}", size = 16)
        plt.xticks(size = 14)
        plt.yticks(ticks = ticks[c], size = 14)
    plt.tight_layout()
    plt.show();

# Evaluation Graphs

def roc_curve(model_prob, X_test, y_test, y_predicted, title, dim, roc_color = "darkorange", baseline_color = "darkblue"):
    """
    Parameters:
    -----------
    model_prob     : the model used for prediction        :     :
    X_test         : the X values                         :     :
    y_test         : true y values                        :     :
    y_predicted    : the model predictions                :     :
    title          : title of the graph                   : str :
    dim            : tuple of the dimensions of the graph : int :
    roc_color      : color value of the ROC curve         : str :
    baseline_color : color value of the baseline          : str :

    Descriptions:
    -------------
    Plots a Receiver Operating Characteristic for a model and includes the AUROC score in the title.

    Returns:
    --------
    Creates a ROC graph for a given model's predictions and allows for appearance control.

    Credit:
    -------
    This code was modified from code written by Matt Brems during our lesson on classification metrics.
    """
    model_prob = [i[0] for i in model_prob.predict_proba(X_test)]
    model_pred_df = pd.DataFrame({"true_values": y_test, "pred_probs": model_prob})
    thresholds = np.linspace(0, 1, 500) 
    def true_positive_rate(df, true_col, pred_prob_col, threshold):
        true_positive = df[(df[true_col] == 1) & (df[pred_prob_col] >= threshold)].shape[0]
        false_negative = df[(df[true_col] == 1) & (df[pred_prob_col] < threshold)].shape[0]
        return true_positive / (true_positive + false_negative)
    def false_positive_rate(df, true_col, pred_prob_col, threshold):
        true_negative = df[(df[true_col] == 0) & (df[pred_prob_col] <= threshold)].shape[0]
        false_positive = df[(df[true_col] == 0) & (df[pred_prob_col] > threshold)].shape[0]
        return 1 - (true_negative / (true_negative + false_positive))
    tpr_values = [true_positive_rate(model_pred_df, "true_values", "pred_probs", prob) for prob in thresholds]
    fpr_values = [false_positive_rate(model_pred_df, "true_values", "pred_probs", prob) for prob in thresholds]
    plt.figure(figsize = dim, facecolor = "white")
    plt.plot(fpr_values, tpr_values, color = roc_color, label = "ROC Curve")
    plt.plot(np.linspace(0, 1, 500), np.linspace(0, 1, 500), color = baseline_color, label = "Baseline")
    rocauc_score = round(roc_auc_score(y_test, y_predicted), 5)
    plt.title(f"{title} With A Score of {rocauc_score}", fontsize = 18)
    plt.ylabel("Sensitivity", size = 16)
    plt.xlabel("1 - Specificity", size = 16)
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    plt.legend(bbox_to_anchor = (1.04, 1), loc = "upper left", fontsize = 16)
    plt.tight_layout()

def residualplots(df, columns, x, dim, titles, row, col, xlabel = "Actual", ylabel = "Predicted"):
    """
    Parameters:
    -----------
    df      : dataframe source of residuals      : dataframe : 
    columns : list of the predicted columns      : str       :
    x       : the actual values                  : str       :
    dim     : tuple of each plot's dimensions    : int       :
    titles  : list of titles for each plot       : str       :
    row     : how many rows will be generated    : int       :
    col     : how many columns will be generated : int       :
    xlabel  : label of the x-axis                : str       :
    ylabel  : label of the y-axis                : str       :

    Description:
    ------------
    This function is designed to be used with a dataframe of the residuals. 
    
    It plots the actual y-values on the x-axis and the predicted on the y-axis.

    Returns:
    --------
    n number of residual plots arranged by the rows and columns.
    """
    count = 0
    fig   = plt.figure(figsize = dim, facecolor = "white")
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        plt.title(f"{titles[c]}", size = 18)
        sns.residplot(x = x, y = column, data = df)
        plt.xlabel(f"{xlabel}", size = 16)
        plt.ylabel(f"{ylabel}", size = 16)
        plt.xticks(size = 14)
        plt.yticks(size = 14)
    plt.tight_layout();
    plt.show();
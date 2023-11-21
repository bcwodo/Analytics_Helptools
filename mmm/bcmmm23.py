import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_growth(n, range=0.2):
    vector = np.linspace(-.5, .5, n)
    result = range/2 * np.sin(np.pi * vector) + 1
    return result

def saturation_hill(x, alpha, gamma): 
    x_s_hill = x ** alpha / (x ** alpha + gamma ** alpha)
    return x_s_hill

def adstock_geometric(x: float, theta: float):
    x_decayed = np.zeros_like(x)
    x_decayed[0] = x[0]
    for xi in range(1, len(x_decayed)):
        x_decayed[xi] = x[xi] + theta* x_decayed[xi - 1]
    return x_decayed

def normalize(a):
    b = (a - np.min(a))/np.ptp(a)
    return b
    
def max_to_one(a):
    m = np.max(a)
    b = a/m
    return b

def transform(x, alpha, gamma, theta, grow=True, growth_range=0.2):
    temp1 = normalize(x)
    temp2 = adstock_geometric(temp1, theta)
    temp3 = saturation_hill(temp2, alpha, gamma)
    if grow==True:
        temp3 = temp3 * get_growth(n=len(x), range=growth_range)
    return temp3


def transform_long(param_df, data_df, variable="L4L5", channel="bm"):
    for v in param_df[variable].values:
      alpha = param_df.loc[param_df[variable] == v, "alpha"].values[0]
      gamma = param_df.loc[param_df[variable] == v, "gamma"].values[0]
      theta = param_df.loc[param_df[variable] == v, "theta"].values[0]
      input_vector = data_df.loc[data_df[variable] == v, "impressions"].values
      output_vector = transform(input_vector, alpha=alpha, gamma=gamma, theta=theta, grow=False)
      data_df.loc[data_df[variable] == v, f"impressions_shaped_{channel}"] = output_vector
    return data_df
    
def transform_long_only_hill(param_df, data_df, variable="L5"):
    for v in param_df[variable].values:
      data_df.loc[data_df[variable] == v, "impressions_to_one"] = max_to_one(data_df.loc[data_df[variable] == v, "impressions"].values)
      alpha = param_df.loc[param_df[variable] == v, "alpha"].values[0]
      gamma = param_df.loc[param_df[variable] == v, "gamma"].values[0]
      input_vector = data_df.loc[data_df[variable] == v, "impressions_to_one"].values
      output_vector = saturation_hill(input_vector, alpha=alpha, gamma=gamma)
      data_df.loc[data_df[variable] == v, "impressions_shaped"] = output_vector
    return data_df


def transform_from_robyn(param, data):
    for v in param.variabel.values:
    # Get Params
        imp_var = f"impressions_{v}"
        alpha = param.loc[param.variabel == v, "alpha"].values[0]
        gamma = param.loc[param.variabel == v, "gamma"].values[0]
        theta = param.loc[param.variabel == v, "theta"].values[0]
        # Shape feature
        out = transform(data[imp_var], alpha=alpha, gamma=gamma, theta=theta)
        data[f"shaped_{v}"] = out

def plot_dual_y(df, column1, column2, label1=None, label2=None):
    """
    Plot two columns from a DataFrame on the same Matplotlib plot with separate y-axes.
    
    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column1 (str): The name of the first column to be plotted.
        column2 (str): The name of the second column to be plotted.
        label1 (str, optional): Label for the first column (default: None).
        label2 (str, optional): Label for the second column (default: None).
    """
    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Plot the first column using the primary y-axis (ax1)
    ax1.plot(df.index, df[column1], color='b', label=label1 or column1)
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel(label1 or column1, color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a secondary y-axis
    ax2 = ax1.twinx()

    # Plot the second column using the secondary y-axis (ax2)
    ax2.plot(df.index, df[column2], color='r', label=label2 or column2)
    ax2.set_ylabel(label2 or column2, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Show legend for both lines
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9))

    plt.title(f'{column1} and {column2} with Separate Y-Axes')
    plt.show()

def mmm_optimizer(param, data):
    for v in param.variabel.values:
        # Get Params
        shape_var = data[f"shaped_{v}"]
        spend_var = data[f"spend_{v}"]
        performance = param.loc[param.variabel == v, "performance"].values[0]
        area_contribution = spend_var.sum() * performance
        area_shapevar = shape_var.sum()
        data[f"contribution_{v}"] = shape_var / area_shapevar * area_contribution


def mmm_optimizer_channel(param, data):
    for v in param.variabel.values:
        # Get Params
        shape_var = data[f"shaped_{v}"]
        spend_var = data[f"spend_{v}"]
        performance = param.loc[param.variabel == v, "performance"].values[0]
        # Add Noise to split
        # sigma = 0.05
        mu = param.loc[param.variabel == v, "split"].values[0]
        split = mu # + sigma * np.random.randn(len(spend_var))
        area_contribution = spend_var.sum() * performance
        area_shapevar = shape_var.sum()
        data[f"contribution_{v}"] = shape_var / area_shapevar * area_contribution
        data[f"bm_contribution_{v}"] = data[f"contribution_{v}"] * split
        data[f"oa_contribution_{v}"] = data[f"contribution_{v}"] - data[f"bm_contribution_{v}"]
        
def mmm_optimizer_long(param_df, data_df, variable="L4L5", channel="bm", exact=False, seed=123, grow=False):
    np.random.seed(seed)
    for v in param_df[variable].values:
        # Get Params
        shape_var = data_df.loc[data_df[variable] == v, f"impressions_shaped_{channel}"].values
        spend_var = data_df.loc[data_df[variable] == v, "spendings"].values
        rho = param_df.loc[param_df[variable] == v, "rho"].values[0]
        tau = param_df.loc[param_df[variable] == v, "tau"].values[0]
        if grow == True:
            delta = param_df.loc[param_df[variable] == v, "delta"].values[0]
            delta_mult = get_growth(len(shape_var), range=delta)
        else:
            delta_mult = np.repeat(1, len(shape_var))
        # Calculation
        if exact == True or rho ==0:
            area_contribution = spend_var.sum() * rho
        else:
            area_contribution = spend_var.sum() * (rho + np.random.rand()/100)
        area_shapevar = shape_var.sum()

        if channel == "bm":
            data_df.loc[data_df[variable] == v, f"{channel}_contribution"] = (shape_var / area_shapevar * area_contribution) * tau * delta_mult
        elif channel == "oa":
            data_df.loc[data_df[variable] == v, f"{channel}_contribution"] = (shape_var / area_shapevar * area_contribution) * (1-tau) * delta_mult
    return data_df


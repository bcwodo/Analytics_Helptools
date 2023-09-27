import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt


kantar_ad_stock_guide = {"banner": [0.45, 0.60],
                         "tv": [0.8,0.9],
                         "search": [0.45,0.60],
                         "olv": [0.6,0.8],
                         "social": [0.3,0.45]}

def saturation_hill(x, alpha, gamma): 
    x_s_hill = x ** alpha / (x ** alpha + gamma ** alpha)
    return x_s_hill

def adstock_geometric(x: float, alpha: float):
    x_decayed = np.zeros_like(x)
    x_decayed[0] = x[0]
    for xi in range(1, len(x_decayed)):
        x_decayed[xi] = x[xi] + alpha* x_decayed[xi - 1]
    return x_decayed

def normalize(a):
    b = (a - np.min(a))/np.ptp(a)
    return b

def transform(x, adstock=0.5, hill=0.5):
    #temp1 = normalize(x)
    temp1 = adstock_geometric(x, adstock)
    temp2 = saturation_hill(temp1, 2, hill)
    return temp2

def norm_transform(x, adstock=0.5, hill=0.5):
    temp1 = normalize(x)
    temp2 = adstock_geometric(temp1, adstock)
    temp3 = saturation_hill(temp2, 2, hill)
    return temp3

def shap_search(df, target, media_features, mtype, log = False, anz_epoch=10):
    """ 
    Sucht die optimalen AdStocks und Gamma für Hill Funktion
    mtype könnne ["tv", "olv", "social", "banner", "search] sein
    """
    # Preprocessing
    y = normalize(df[target]).values
    X_unscaled = df[media_features].values
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X_unscaled)
    X_base = np.copy(X)

    # Grid Search
    params = {}
    for epoche in range(anz_epoch):
        for i in range(len(media_features)):
            # Grids
            mt = mtype[i]
            adst_grid = np.linspace(kantar_ad_stock_guide[mt][0], kantar_ad_stock_guide[mt][1], 10)
            gamma_grid = np.linspace(0.3,0.9, 10)

            
            model = LinearRegression()
            
            bench = 0
            for adst in adst_grid:
                for gamma in gamma_grid:
                    X[:,i] = transform(X_base[:,i], adstock=adst, hill=gamma)
                    model.fit(X,y)
                    r2_mod = r2_score(y, model.predict(X))
                    if r2_mod>bench:
                        bench = r2_mod
                        best_adst = adst
                        best_gamma = gamma
                        params.update({media_features[i]: [best_adst, best_gamma]})
                        
            X[:,i] = transform(X_base[:,i], adstock=best_adst, hill=best_gamma)
            if log:
                print(f"Epoche: {epoche}  Media: {media_features[i]}, AdStock: {best_adst:.2f}, Gamma: {best_gamma:.2f}, R² {bench:.2f}")
    return pd.DataFrame(params, index=["AdStock", "Gamma"]).T


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

def modell_contribution(df, impression_var, shape_var, robyn_value):
    """ Modelliert die Shape Variable hinsichtilch der 'robyn' Ergebnisse"""
    sum_impressions = df[impression_var].sum()
    sum_shape = df[shape_var].sum()
    effekt = robyn_value
    multiplier = (effekt * sum_impressions) / sum_shape
    contribution = df[shape_var]*multiplier
    return contribution

def make_contribution_var(df, impression_var, shape_var, robyn_value, prefix="Contribution_"):
    temp = df.copy()
    for i in range(len(impression_var)):
        temp[f"{prefix}{impression_var[i]}"] = modell_contribution(df=df, impression_var=impression_var[i], shape_var=shape_var[i], robyn_value=robyn_value[i])
    outvar = [c for c in temp.columns if c.startswith(prefix)]
    return temp[outvar]

def goodness_of_fit(target_df, features_df):
    target_mean = target_df.mean()
    feature_sum = features_df.sum(axis=1)
    feature_mean = feature_sum.mean()
    y = target_df - target_mean
    x = feature_sum - feature_mean
    delta = np.sum(np.abs(y-x))
    return delta

def modell_optimizer(df, target, impression_var, shape_var, robyn_value, runs = 10):
    bench = np.inf
    boost = np.linspace(0.9,1.1,10)
    for lauf in range(runs):
        mult = np.random.choice(boost, size=len(impression_var), replace=True)
        rob_mult = robyn_value*mult
        temp = make_contribution_var(df=df, impression_var=impression_var, shape_var=shape_var, robyn_value=rob_mult)
        gof = goodness_of_fit(df[target], temp)
        if gof < bench:
            bench = gof
            best_rob = rob_mult
            out = temp
    return out, best_rob


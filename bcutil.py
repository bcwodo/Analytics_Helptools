import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

def cor2rel(C, threshold=0.5):
    nvar = C.shape[0]
    rel = []
    for i in range(0,nvar-1):
        for j in range(i+1, nvar):
            a = C.columns[i]
            b = C.columns[j]
            r = C.iloc[i, j] 
            if r > threshold:
                rel.append((a,b,1))
            else:
                pass
           #rel.append((a,b,0)) 
    rel_df = pd.DataFrame(rel)
    rel_df.columns = ["Source", "Target", "Value"]
    return rel_df


def cor2rel_part(C, numcol, threshold=0.5):
    nvar = C.shape[0]
    rel = []
    for i in range(0, numcol):
        for j in range(i+1, nvar):
            a = C.columns[i]
            b = C.columns[j]
            r = C.iloc[i, j] 
            if r > threshold:
                rel.append((a,b,1))
            else:
                pass
           #rel.append((a,b,0)) 
    rel_df = pd.DataFrame(rel)
    rel_df.columns = ["Source", "Target", "Value"]
    return rel_df


def weight_df(df, weight):
    all_weights = df[weight].unique()
    min_weight = all_weights.min()

    w_df_list = []
    for w in all_weights:
        wfac = round(w/min_weight)
        w_df_list += [df.loc[df[weight]==w] for _ in range(wfac)]
    return pd.concat(w_df_list)


def generate_heatmap(data, x_labels=None, y_labels=None, title=None, plotsize=(6, 4), show_legend=False, legend_title=None):
    """
    Generate and display a heatmap.

    Parameters:
        data (numpy.ndarray): The 2D array containing the data to be plotted as a heatmap.
        x_labels (list or None): List of labels for the x-axis ticks. If None, x-axis ticks are not labeled.
        y_labels (list or None): List of labels for the y-axis ticks. If None, y-axis ticks are not labeled.
        title (str or None): The title of the heatmap. If None, no title is displayed.
        plotsize (tuple): The size of the heatmap plot in inches (width, height). Default is (10, 6).
        show_legend (bool): Whether to display the colorbar/legend. Default is False.
        legend_title (str or None): Optional title for the colorbar/legend. If None, no title is displayed.

    Returns:
        None

    Example:
        a = [1, 2, 3, 4]
        p = [10, 20, 30]
        M = np.outer(p, a)  # Outer product of 'a' and 'p'

        # Customize labels and title
        x_labels = ['A', 'B', 'C', 'D']
        y_labels = ['P1', 'P2', 'P3']
        heatmap_title = "Heatmap of Outer Product"
        legend_title = "Values of Outer Product"

        # Generate and display the heatmap with optional parameters
        plotsize = (12, 8)  # Set the desired size of the heatmap
        show_legend = True  # Set to False if you don't want to show the colorbar
        generate_heatmap(M, x_labels=x_labels, y_labels=y_labels, title=heatmap_title,
                         plotsize=plotsize, show_legend=show_legend, legend_title=legend_title)
    """
    fig, ax = plt.subplots(figsize=plotsize)
    im = ax.imshow(data, cmap='viridis')

    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    # Loop over data dimensions and create text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, f'{data[i, j]:.2f}', ha="center", va="center", color="w")

    if title is not None:
        ax.set_title(title)

    if show_legend:
        if legend_title is not None:
            cbar = ax.figure.colorbar(im, ax=ax, label=legend_title)
        else:
            cbar = ax.figure.colorbar(im, ax=ax)
    else:
        cbar = None

    if cbar:
        cbar.ax.set_ylabel(legend_title, rotation=-90, va="bottom")

    plt.show()



def generate_matrix_from_function(vector1, vector2, math_function):
    """
    Generate a matrix by applying the mathematical function to corresponding elements of two input vectors.

    Parameters:
        vector1 (list or numpy.ndarray): First input vector.
        vector2 (list or numpy.ndarray): Second input vector.
        math_function (function): A mathematical function that takes two arguments.

    Returns:
        numpy.ndarray: A matrix where each element is the result of applying the mathematical function
                       to the corresponding elements of the input vectors.

    Example:
        def add(x, y):
            return x + y

        a = [1, 2, 3]
        b = [4, 5, 6, 7]
        result_matrix = generate_matrix_from_function(a, b, add)
        print(result_matrix)
        # Output: [[ 5.  6.  7.  8.]
        #          [ 6.  7.  8.  9.]
        #          [ 7.  8.  9. 10.]]
    """
    # Convert input vectors to numpy arrays (if not already)
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Use broadcasting to apply the mathematical function to the corresponding elements of the input vectors
    result_matrix = math_function(vector1[:, np.newaxis], vector2)

    return result_matrix


def sw(string, df):
    return [c for c in df.columns if c.startswith(string)]



import polars as ps
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(df, col1, bins=30):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Single histogram
    ax.hist(df[col1].to_numpy(), bins=bins, alpha=0.7, edgecolor='black', color='skyblue')
    ax.set_title(f'{col1} Distribution')
    ax.set_xlabel(col1)
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

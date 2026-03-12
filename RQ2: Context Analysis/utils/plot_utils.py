"""
Plotting utilities and style configuration
"""

import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style(figsize: tuple = (40, 24),
                    font_size: int = 24,
                    title_size: int = 32,
                    label_size: int = 28,
                    tick_size: int = 20,
                    legend_size: int = 20,
                    figure_title_size: int = 36):
    """
    Set up plotting style for large, clear plots
    
    Args:
        figsize: Figure size tuple
        font_size: Base font size
        title_size: Title font size
        label_size: Axis label font size
        tick_size: Tick label font size
        legend_size: Legend font size
        figure_title_size: Figure title font size
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.titlesize'] = title_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['legend.fontsize'] = legend_size
    plt.rcParams['figure.titlesize'] = figure_title_size


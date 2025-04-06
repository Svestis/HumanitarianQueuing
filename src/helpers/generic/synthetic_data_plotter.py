"""
synthetic_data_plotter.py

This module generates descriptive statistics and visualizations from a synthetic dataset,
including histograms, count plots, and arrival time distributions. It outputs results to
PNG files and appends them to a Word document for reporting purposes.

Last Updated: 2025_04_06
"""
import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.helpers.generic.word_generator import append_images, append_table_from_df
from typing import Optional, Callable, Any
import re

matplotlib.use('Agg')  # Non GUI - save to file

OUTPUT_DIR = "./src/resources/synthetic_data_summary/"

def save_plot(title: str, output_path: str = OUTPUT_DIR) -> str:
    """
     Helper function to plot & save
    :param title: The title of the graph
    :type title: str
    :param output_path: The save directory
    :type output_path: str
    :return: The file_path
    :rtype: str
    """
    title_cleansed: str = re.sub(r'\W+', '', title)
    file_name: str = os.path.join(output_path, f"{title_cleansed}.png")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf(); plt.close()

    return file_name

def plot_and_save(plot_function: Callable[[], Any], title: str, output_path: str = OUTPUT_DIR,
                  xlabel: Optional[str] = None, ylabel: Optional[str] = "Count") -> str:
    """
    Generalising plotting and running save function
    :param plot_function: The function of the specific plot (defined further down)
    :type plot_function: Callable[[],None]
    :param title: The title of the plot used also in the save path
    :type title: str
    :param output_path: The dir were plots will be saved, used along with title for the full path
    :type output_path: str
    :param xlabel: The text of the xlabel
    :type xlabel: str
    :param ylabel: The text of the ylabel
    :type ylabel: str
    :return: The path after running save_plot
    :rtype: str
    """
    plt.figure(figsize=(8, 5))

    #Running the specific plot function
    plot_function()

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    return save_plot(title, output_path)

def plot_histogram(df: pd.DataFrame, column: str, title: str, xlabel: str, ylabel: Optional[str] = "Count",
                   output_path: Optional[str] = OUTPUT_DIR, bins: Optional[int] = 20,
                   kde: Optional[bool] = True)\
        -> tuple[str, str]:
    """
    Generic function to create histograms based on input data, parameterizing basic variables

    :param df: The dataframe that will be used to plot data (mostly sourced from synthetic_data.csv)
    :type df: pd.DataFrame
    :param column: The column of the dataframe to plot
    :type column: str
    :param title: The title of the plot, used also for file naming purposes
    :type title: str
    :param xlabel: The label of the X axis of the plot
    :type xlabel: str
    :param ylabel: The label of the y axis of the plot
    :type ylabel: str
    :param output_path: The output path (folder-only)
    :type output_path: str
    :param bins: The number of bins used in a histogram, defaulting to 20
    :type bins: int
    :param kde: Display curve only in case of continuous distribution
    :type kde: bool
    :return: Tuple[str, str] — the output path and plot title.
    :rtype: tuple[str, str]
    """
    if bins is None:
        bins = min(50, int(len(df[column]) ** 0.5))  # smart binning

    file_name: str = plot_and_save(lambda: sns.histplot(df[column], bins=bins, kde=kde), title, output_path,
                                   xlabel, ylabel)
    return file_name, title

def plot_count(df: pd.DataFrame, column: str, title: str, xlabel: str, ylabel: Optional[str] = "Count",
               output_path: Optional[str] = OUTPUT_DIR) -> tuple[str, str]:
    """
    Generic function to create bar plots based on input data, parameterizing basic variables

    :param df: The dataframe that will be used to plot data (mostly sourced from synthetic_data.csv)
    :type df: pd.DataFrame
    :param column: The column of the dataframe to plot
    :type column: str
    :param title: The title of the plot, used also for file naming purposes
    :type title: str
    :param xlabel: The label of the X axis of the plot
    :type xlabel: str
    :param ylabel: The label of the y axis of the plot
    :type ylabel: str
    :param output_path: The output path (folder-only)
    :type output_path: str
    :return: Tuple[str, str] — the output path and plot title.
    :rtype: tuple[str, str]
    """
    file_name = plot_and_save(lambda: sns.countplot(x=column, data=df), title, output_path, xlabel)

    return file_name, title

def plot_arrival_density(df: pd.DataFrame, title: str = "Arrival Time Density",
                         output_path: Optional[str] = OUTPUT_DIR, xlabel: Optional[str] = "Minute of the day",
                         ylabel: Optional[str] = "Number of Arrivals",
                         column: Optional[str] = "ArrivalTime") -> tuple[str, str]:
    """
    Specific function to plot arrival density
    :param df: The dataframe that will be used to plot data (mostly sourced from synthetic_data.csv)
    :type df: pd.DataFrame
    :param title: The title of the plot, used also for file naming purposes
    :type title: str
    :param xlabel: The label of the X axis of the plot
    :type xlabel: str
    :param ylabel: The label of the y axis of the plot
    :type ylabel: str
    :param output_path: The output path (folder-only)
    :type output_path: str
    :param column: The column of the dataframe to be used
    :type column: str
    :return: Tuple[str, str] — the output path and plot title.
    :rtype: tuple[str, str]
    """
    df['ArrivalTimeFull'] = pd.to_datetime(df[column], format='%H:%M')
    df = df.sort_values(by='ArrivalTimeFull')
    df['Minute'] = df['ArrivalTimeFull'].dt.hour * 60 + df['ArrivalTimeFull'].dt.minute

    file_path = plot_and_save(lambda: sns.histplot(df['Minute'], bins=60, kde=True), title, output_path, xlabel, ylabel)

    return file_path, title

def plot_dataset(file_path: Optional[str]="./resources/data_generator/synthetic_data.csv",
                 df: Optional[pd.DataFrame] = None) -> None:
    """
    Plotting the graphs based on the provided dataset and calling a word-writing function

    :param df: Passing a dataframe as an alternative to the path
    :type df: Optional[pd.Dataframe]
    :param file_path: The path of the synthetic data file
    :type file_path: str
    :return: None
    :raises ValueError: If file_path and df are bot none
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

    if df is None:
        df: pd.DataFrame = pd.read_csv(file_path)

    if df is None and file_path is None:
        raise ValueError("Either 'df' or 'file_path' must be provided.")

    images_path: list[str] = []
    captions: list[str] = []

    # Converting arrival time to hours
    df['ArrivalHour'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M').dt.hour

    # Summary statistics
    summary: pd.DataFrame = df[['Age', 'TransactionTime', 'VulnerabilityScore']].describe()
    summary = summary.reset_index().rename(columns={'index': 'Statistic'})
    append_table_from_df(summary, heading="Summary of synthetic data")

    # Defining plots
    plots: list = [
        (plot_histogram, df, 'Age', "Age Distribution", "Age"),
        (plot_histogram, df, 'TransactionTime', "Transaction Time Distribution", "Transaction Time (minutes)"),
        (plot_histogram, df, 'VulnerabilityScore', "Vulnerability Score Distribution", "Vulnerability Score"),
        (plot_count, df, 'Gender', "Gender Distribution", "Gender"),
        (plot_histogram, df, "ArrivalHour", "Arrival Time Distribution by Hour", "Hour of Arrival",
         "Number of Arrivals", OUTPUT_DIR, 10, False),
        (plot_arrival_density,df, "Arrival Time Density")
    ]

    # Iteratively run functions for plotting
    for func, *args in plots:
        path, title = func(*args)
        images_path.append(path)
        captions.append(title)

    append_images(images_path, captions=captions, heading="Synthetic Data Graphs")
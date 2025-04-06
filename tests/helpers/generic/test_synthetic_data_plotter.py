"""
test_synthetic_data_plotter.py

Validating synthetic_data_plotter.py functionality

Last Updated: 2025_04_06
"""
import os
import pathlib
from pathlib import Path
import pandas as pd
import pytest
import shutil
from tempfile import mkdtemp
from helpers.generic.synthetic_data_plotter import plot_histogram, plot_count, plot_dataset, OUTPUT_DIR
from typing import Generator
from helpers.generic import synthetic_data_plotter

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Creating a sample df for testing plots.

    :return: A small dataframe for testing purposes
    :rtype: pd.Dataframe
    """
    return pd.DataFrame({
        'Age': [25, 45, 60, 30],
        'TransactionTime': [10, 15, 12, 8],
        'VulnerabilityScore': [0.2, 0.5, 0.8, 0.3],
        'Gender': ['Male', 'Female', 'Female', 'Other'],
        'ArrivalTime': ['08:00', '08:30', '09:00', '09:30']
    })

@pytest.fixture
def tmp_output_dir() -> Generator[str, None, None]:
    """
    Creates and returns a tmporary directory for plot file output.

    :return: Path to a tmporary directory for use in test functions.
    :rtype: Generator
    """
    dir_path: str = mkdtemp()

    yield dir_path

    shutil.rmtree(dir_path)

def test_plot_histogram_creates_file(sample_df: pd.DataFrame, tmp_output_dir: str) -> None:
    """
    Test whether plot_histogram generates a PNG file and correct title.

    :param sample_df: Provided by fixture.
    :type sample_df: pd.DataFrame
    :param tmp_output_dir: tmporary directory fixture.
    :type tmp_output_dir: str
    :return: None
    """
    out_path, title = plot_histogram(
        df=sample_df,
        column='Age',
        title='Test Age Dist',
        xlabel='Age',
        output_path=tmp_output_dir
    )

    expected_file: str = os.path.join(tmp_output_dir, "TestAgeDist.png")

    assert os.path.isfile(expected_file)
    assert title == 'Test Age Dist'

def test_plot_count_creates_file(sample_df: pd.DataFrame, tmp_output_dir: str) -> None:
    """
    Test whether plot_histogram generates a PNG file and correct title.

    :param sample_df: Provided by fixture.
    :type sample_df: pd.DataFrame
    :param tmp_output_dir: tmporary directory fixture.
    :type tmp_output_dir: str
    :return: None
    """
    out_path, title = plot_count(
        df=sample_df,
        column='Gender',
        title='Gender Count',
        xlabel='Gender',
        output_path=tmp_output_dir
    )

    expected_file: str = os.path.join(tmp_output_dir, "GenderCount.png")

    assert os.path.isfile(expected_file)
    assert title == 'Gender Count'

def test_plot_dataset_runs_without_error(tmp_path: pathlib.Path) -> None:
    """
    Test whether plot_dataset executes successfully with minimal valid data.

    :param tmp_path: Built-in pytest fixture providing a tmporary path.
    :type tmp_path: pathlib.Path
    :return: None
    """
    test_csv: Path = tmp_path / "test_data.csv"
    df: pd.DataFrame = pd.DataFrame({
        'Age': [30, 40],
        'TransactionTime': [5, 10],
        'VulnerabilityScore': [0.1, 0.9],
        'Gender': ['Male', 'Female'],
        'ArrivalTime': ['08:00', '08:15']
    })

    df.to_csv(test_csv, index=False)
    plot_dataset(file_path=str(test_csv))

    assert os.path.isdir("./resources/synthetic_data_summary")

def test_plot_histogram_raises_with_invalid_column(sample_df: pd.DataFrame, tmp_output_dir: pathlib.Path)\
        -> None:
    """
    Checking for missing column edge case
    :param sample_df: Provided by fixture.
    :type sample_df: pd.DataFrame
    :param tmp_output_dir: tmp dir for output
    :type tmp_output_dir: pathlib.Path
    :return: None
    """
    with pytest.raises(KeyError):
        plot_histogram(
            df=sample_df,
            column='NonExistent',
            title='Bad Column',
            xlabel='Bad',
            output_path=os.path.join(tmp_output_dir, "bad_column.png")
        )


def test_plot_dataset_with_dataframe(tmp_output_dir: pathlib.Path = OUTPUT_DIR) -> None:
    """
    Test that plot_dataset works when only a DataFrame is passed.
    :param tmp_output_dir: tmporary directory provided by fixture
    :type tmp_output_dir: str
    :return: None
    """
    df: pd.DataFrame = pd.DataFrame({
        'Age': [30, 40],
        'TransactionTime': [5, 10],
        'VulnerabilityScore': [0.2, 0.9],
        'Gender': ['Male', 'Female'],
        'ArrivalTime': ['08:00', '08:15']
    })

    synthetic_data_plotter.plot_dataset(df=df)

    expected: Path = Path(tmp_output_dir) / "GenderDistribution.png"
    print(tmp_output_dir)
    assert expected.exists()


def test_plot_dataset_missing_input() -> None:
    """
    Test that plot_dataset raises a ValueError when neither file_path nor df is provided
    """
    with pytest.raises(ValueError):
        plot_dataset(file_path=None, df=None)


def test_plot_dataset_file_path_precedence(tmp_output_dir: Path, tmp_path: Path) -> None:
    """
    Ensure file_path overrides df
    :param tmp_output_dir: tmp directory
    :type tmp_output_dir: Path
    :param tmp_path: tmp files
    :type tmp_path: Path
    :return: None
    """
    csv_path: Path = tmp_path / "input.csv"

    df: pd.DataFrame = pd.DataFrame({
        'Age': [22, 33],
        'TransactionTime': [6, 12],
        'VulnerabilityScore': [0.3, 0.7],
        'Gender': ['Male', 'Female'],
        'ArrivalTime': ['08:00', '08:30']
    })
    df.to_csv(csv_path, index=False)

    synthetic_data_plotter.OUTPUT_DIR = str(tmp_output_dir)

    wrong_df = df.copy()
    wrong_df['Age'] = [100, 200]

    synthetic_data_plotter.plot_dataset(file_path=str(csv_path), df=wrong_df)

    output = pd.read_csv(csv_path)
    assert 100 not in output['Age'].values
"""
test_data_generator.py

Validating data_generator.py functionality

Last Updated: 2025_04_06
"""
from datetime import datetime
import _pytest.monkeypatch
import pytest
import pandas as pd
from src.helpers.generic.data_generator import load_config, validate_config, data_generator, CONFIG_MAP, save
from pathlib import Path

def test_load_config_valid() -> None:
    """
    Test that the default configuration loads successfully.

    Ensures that the returned configuration is a dictionary with required keys.
    :return: None
    """
    config: dict = load_config("default")

    assert isinstance(config, dict) # Ensuring load_config returns a dict
    assert "columns" in config
    assert config["num_records"] > 0

def test_validate_config_complete() -> None:
    """
    Test that a valid configuration passes validation without errors.
    :return: None
    """
    config: dict = load_config("default")
    validate_config(config)  # Should not raise any AssertionError

def test_validate_config_missing_column() -> None:
    """
    Test that missing a required column in the configuration raises an AssertionError.
    :return: None
    """
    config: dict = load_config("default")

    del config["columns"]["age"]

    with pytest.raises(AssertionError) as excinfo:
        validate_config(config)

    assert "Missing 'age'" in str(excinfo.value)


def test_data_generator_output(tmp_path: Path) -> None:
    """
    Test that the data generator writes a file and returns a valid DataFrame + generated CSV matches the DataFrame.
    :param tmp_path: The path for the tmp files
    :type tmp_path: Path
    :return: None
    """
    output_file: Path = tmp_path / "synthetic_test.csv"
    df: pd.DataFrame = save(config_type="default", file_path=str(output_file))

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert output_file.exists()

    loaded: pd.DataFrame = pd.read_csv(output_file)

    assert loaded.shape == df.shape
    assert set(df.columns) == set(loaded.columns)


def test_data_generator_structure() -> None:
    """
    Test that the generated DataFrame has all expected columns.
    :return: None
    """
    df: pd.DataFrame = save(config_type="default")
    expected_columns: list = ['ID', 'Gender', 'Age', 'TransactionTime', 'VulnerabilityScore', 'ArrivalTime']

    assert all(col in df.columns for col in expected_columns)


def test_value_ranges() -> None:
    """
    Test that generated numerical values are within min/max bounds.
    :return: None
    """
    df: pd.DataFrame = save(config_type="default")
    config: dict = load_config("default")

    age_cfg: pd.IndexSlice = config['columns']['age']['params']
    assert df['Age'].between(age_cfg['left'], age_cfg['right']).all()

    tx_cfg: pd.IndexSlice = config['columns']['transaction_time']['params']
    assert df['TransactionTime'].between(tx_cfg['left'], tx_cfg['right']).all()

    assert df['VulnerabilityScore'].between(0, 1).all()  # b-dist is always between 0 & 1, needed if dist is changed


def test_gender_distribution() -> None:
    """
    Check that gender distribution in the generated dataset is statistically reasonable.
    :return: None
    """
    df: pd.DataFrame = save("default")
    config: dict = load_config("default")
    gender_config: pd.IndexSlice = config['columns']['gender']

    # Calculate the actual distribution
    value_counts: dict = df['Gender'].value_counts(normalize=True).to_dict()

    # Compare actual distribution with expected + 10% error margin
    for gender, expected_prob in zip(gender_config['values'], gender_config['probabilities']):
        assert abs(value_counts.get(gender, 0) - expected_prob) < 0.1


def test_arrival_time_format() -> None:
    """
    Ensure all arrival times are valid and within expected time window.
    :return: None
    """
    df: pd.DataFrame = save(config_type="default")
    for time_str in df['ArrivalTime']:
        t: datetime = datetime.strptime(time_str, "%H:%M")
        assert 0 <= t.hour < 24 and 0 <= t.minute < 60


def test_unique_ids() -> None:
    """
    Verify that all generated IDs are unique.
    :return: None
    """
    df: pd.DataFrame = save(config_type="default")

    assert df['ID'].is_unique


@pytest.mark.parametrize("config_type", CONFIG_MAP.keys()) # run for each config
def test_config_profiles(config_type: str, tmp_path: Path) -> None:
    """
    Check that the data generator works for all config types.
    :param config_type: Passing each different config type
    :type config_type: str
    :param tmp_path: tmp output path
    :type tmp_path: Path
    :return: None
    """
    output_file: Path = tmp_path / f"{config_type}_test.csv"
    df: pd.DataFrame = save(config_type=config_type, file_path=str(output_file))

    assert output_file.exists()
    assert not df.empty


@pytest.mark.parametrize("config_type",  CONFIG_MAP.keys())
def test_arrival_distribution(config_type: str) -> None:
    """
    Test that the arrival distribution generates a valid set of arrival times.
    :param config_type: Passing each different config type
    :type config_type: str
    :return: None
    """
    df: pd.DataFrame = save(config_type=config_type)
    assert len(df['ArrivalTime']) > 0
    for time_str in df['ArrivalTime']:
        t: datetime = datetime.strptime(time_str, "%H:%M")
        assert 0 <= t.hour < 24


@pytest.mark.parametrize("config_type",  CONFIG_MAP.keys())
def test_data_generator_config_variants(config_type):
    """
    Ensuring that no empty df is created ofr different distributions
    :param config_type: Passing each different config type
    :type config_type: str
    :return: None
    """
    df = save(config_type=config_type)
    assert not df.empty

def test_load_config_missing_file(monkeypatch: _pytest.monkeypatch.MonkeyPatch) -> None:
    """
    Negative test for missing patch
    :param monkeypatch: Pytest fixure
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return:None
    """
    monkeypatch.setitem(CONFIG_MAP, "none", "/nonexistent/config.json")
    with pytest.raises(FileNotFoundError):
        load_config("none")

def test_invalid_arrival_distribution() -> None:
    """
    Testing invalid arrival distribution
    :return: None
    :raises ValueError: If the arrival distribution is not one of the supported types
    """
    config = load_config("default")
    config.setdefault("arrival_time", {})
    config["arrival_time"]["distribution"] = "unsupported_dist"

    with pytest.raises(ValueError, match="Unsupported distribution type"):
        data_generator(config)

def test_column_types() -> None:
    """
    Tests the column types
    :return: None
    """
    df = save("default")
    assert df["ID"].dtype == int
    assert df["Age"].dtype == int
    assert pd.api.types.is_float_dtype(df["TransactionTime"])
    assert pd.api.types.is_float_dtype(df["VulnerabilityScore"])
    assert pd.api.types.is_object_dtype(df["Gender"])
    assert pd.api.types.is_object_dtype(df["ArrivalTime"])
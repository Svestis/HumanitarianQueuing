"""
run_tests.py

Programatically running all tests sequentially

Last Updated: 2025_04_01
"""
import os
import shutil
import pytest

def run_tests() -> None:
    """
    Running tests sequentially
    :return: None
    """
    test_files: list = [
        "helpers/generic/test_data_generator.py",
        "helpers/generic/test_synthetic_data_plotter.py",
        "helpers/generic/test_word_generator.py",
        "helpers/models/test_metrics.py",
        "helpers/models/test_util.py",
        "models/base/test_fifo.py",
        "models/base/test_lifo.py",
        "models/base/test_mmc.py",
        "models/base/test_sjf.py",
        "models/base/test_ros.py",
        "models/priority/test_community.py",
        "models/priority/test_context.py",
        "models/priority/test_fair.py",
        "models/priority/test_fair_prop.py",
        "models/priority/test_priority.py"
    ]

    results = {}

    for test_file in test_files:
        print(f"Running {test_file}...")
        try:
            result = pytest.main([test_file])
            results[test_file] = "PASSED" if result==0 else "FAILED"
        except Exception as e:
            results[test_file] = f"ERROR: {e}"

    print("\nSUMMARY:")
    for test, result in results.items():
        print(f"{test}: {result}")

def cleanup() -> None:
    """Cleaning up folders before re-running tests. New resources remain for validation if needed7"""
    path = "resources"

    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    else:
        pass

if __name__ == "__main__":
    cleanup()
    run_tests()

# TODO: Modul common tests
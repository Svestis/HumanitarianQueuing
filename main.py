from src.helpers.generic.data_generator import save
from src.helpers.generic.synthetic_data_plotter import plot_dataset

original_df = save(config_type="uniform")
plot_dataset(df=original_df)

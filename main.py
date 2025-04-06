from helpers.generic.data_generator import save
from helpers.generic.synthetic_data_plotter import plot_dataset

original_df = save(config_type="uniform")
plot_dataset(df=original_df)

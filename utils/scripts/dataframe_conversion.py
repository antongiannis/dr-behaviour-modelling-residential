import pandas as pd 
import phdTools
from pathlib import Path

notebook_path = Path.cwd()
data_path = notebook_path / "data"
csv_path = data_path / "househould_plot_df.csv"
df = pd.read_csv(csv_path)

unique_vals = df.phd.unique_cols()

dict_keys = list(unique_vals.keys())
dict_values = list(unique_vals.values())

structure_df = pd.DataFrame({'columns': dict_keys, 'col_values':dict_values})
out_csv_path = data_path / 'structure_df.csv'
structure_df.explode('col_values').to_csv(out_csv_path, index=False)
print("Structure table created successfully")

import pandas as pd
import glob, os

out_dir = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\Cainine\PreProcessedData"
files = glob.glob(os.path.join(out_dir, "*_mitbih.csv"))

dfs = [pd.read_csv(f, header=None) for f in files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv(os.path.join(out_dir, "canine_all_mitbih.csv"), header=False, index=False)

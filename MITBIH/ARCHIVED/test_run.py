from test_code import predict_from_kaggle_csv_row, quick_eval, full_file_eval, find_examples_by_true_label

csv_path = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\Cainine\PreProcessedDatav2\canine_all_enhanced.csv"

#csv_path = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\Cainine\PreProcessedData\canine_all_mitbih.csv" # cainine

# Full evaluation (or set limit=5000 for speed)
full_file_eval(csv_path, zscore=True, limit=None)

# Peek at minority classes: try label 1 (S), then 2 (V), 4 (Q)
find_examples_by_true_label(csv_path, label=1, k=10)  # see if/where it predicts S
find_examples_by_true_label(csv_path, label=2, k=10)
find_examples_by_true_label(csv_path, label=4, k=10)

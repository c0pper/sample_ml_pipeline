def load_dataset(dataset_path, x_label, y_label):
    import pandas as pd

    print("\n\n[+] Loading dataset\n\n")
    dataset = pd.read_csv(dataset_path, usecols=[x_label, y_label])
    return dataset
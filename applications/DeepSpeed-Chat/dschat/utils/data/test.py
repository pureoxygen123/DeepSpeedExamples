from raw_datasets import DahoasFullhhrlhfDataset

dataset = DahoasFullhhrlhfDataset()
train_data = dataset.get_train_data()
eval_data = dataset.get_eval_data()
print(train_data)
print(eval_data)
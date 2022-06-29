import sys
if '.' not in sys.path:
    sys.path.insert(0, '.')
sys.setrecursionlimit(3000)
import pandas as pd

def import_model():
    from .tablegan_helper import TableGAN

def train(js, dataset, epochs=200):
    from .tablegan_helper import TableGAN
    from sdgym.datasets import get_dataset_paths, load_dataset, load_tables
    import rdt
    # TODO: last column must be target column :|
    def preprocess_dataset(dataset):
        dataset = dataset.dropna()
        return dataset
    table_name = js['file']
    dataset_root_folder = "datasets/"
    datasets = get_dataset_paths([js['folder']], dataset_root_folder, "", "", "")
    metadata = load_dataset(datasets[0])
    real_data = load_tables(metadata)
    mm = metadata.get_table_meta(table_name)
    data = real_data[table_name]
    data = preprocess_dataset(data)
    vv = data[0:0]
    justthistable = {}
    justthistable[table_name] = data

    additionals = js['additionals']
    # TODO add other options
    gc = TableGAN(epochs=epochs, opacus=additionals("use_dp"))
    columns, categoricals = gc._get_columns(data, mm)
    discrete = list(set(range(len(columns))) - set(categoricals))
    data_cat = data.iloc[:, categoricals]
    data_dis = data.iloc[:, discrete]
    data = pd.concat([data_dis, data_cat], axis=1, join="inner")
    justthistable[table_name] = data
    columns, categoricals = gc._get_columns(data, mm)
    gc.update_column_info(columns)
    ht = rdt.HyperTransformer(dtype_transformers={
        'O': 'label_encoding',
    })
    ht.fit(data.iloc[:, categoricals])
    gc.fit_sample(justthistable.copy(), metadata)
    return gc

def generate(js, model, sample_total=40000):
    # TODO cleanup
    from sdgym.datasets import get_dataset_paths, load_dataset, load_tables
    import rdt
    # TODO: last column must be target column :|
    def preprocess_dataset(dataset):
        dataset = dataset.dropna()
        return dataset
    table_name = js['file']
    dataset_root_folder = "datasets/"
    datasets = get_dataset_paths([js['folder']], dataset_root_folder, "", "", "")
    metadata = load_dataset(datasets[0])
    real_data = load_tables(metadata)
    mm = metadata.get_table_meta(table_name)
    data = real_data[table_name]
    data = preprocess_dataset(data)
    vv = data[0:0]
    justthistable = {}
    justthistable[table_name] = data

    # TODO add other options
    columns, categoricals = model._get_columns(data, mm)
    discrete = list(set(range(len(columns))) - set(categoricals))
    data_cat = data.iloc[:, categoricals]
    data_dis = data.iloc[:, discrete]
    data = pd.concat([data_dis, data_cat], axis=1, join="inner")
    justthistable[table_name] = data
    columns, categoricals = model._get_columns(data, mm)
    model.update_column_info(columns)
    ht = rdt.HyperTransformer(dtype_transformers={
        'O': 'label_encoding',
    })
    ht.fit(data.iloc[:, categoricals])
    s = model.sample(sample_total)
    s = pd.DataFrame(s, columns=columns)
    s.iloc[:, categoricals] = ht.reverse_transform(s.iloc[:, categoricals])
    vv = vv.append(s, ignore_index=True)
    return vv
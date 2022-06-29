def dcr(js, dataset, sample_dataset, logger, error_logger):
    import numpy as np
    import statistics
    import pandas as pd
    import torch
    discrete_columns = js['discrete_columns']
    columns = list(sample_dataset)
    quantile_count = 100
    normalizer = {}
    for column in columns:
        if (column in discrete_columns):
            continue
        data = dataset[column]
        pure = [i/quantile_count for i in range(quantile_count+1)]
        quantiles = np.quantile(data, pure)
        normalizer[column] = quantiles

    def convert_quantiles(value, quantiles):
        for i in range(quantile_count + 1):
            if (value < quantiles[i]):
                return i/quantile_count
        return 1
    
    continuous_columns = list(set(columns) - set(discrete_columns))
    continuous_columns
    for column in continuous_columns:
        quantiles = normalizer[column]
        dataset[column] = dataset[column].apply(convert_quantiles, args=(quantiles,))
        sample_dataset[column] = sample_dataset[column].apply(convert_quantiles, args=(quantiles,))
    
    def calculate_distance(record, dataset):
        dataset_plus = abs(record[continuous_columns] - dataset[continuous_columns])
        dataset_plus = pd.concat([dataset_plus, record[discrete_columns] != dataset[discrete_columns]], axis=1)
        dataset_plus = dataset_plus.apply(np.linalg.norm, axis=1)
        dataset_plus = dataset_plus.sort_values() # type: ignore
        return dataset_plus

    logger.write("[INFO]\tPRIVACY\n")
    vals = []
    valsnth1 = []
    valsnth5 = []
    print(f"Checking privacy of {min(5000, len(sample_dataset))} rows")
    for i in range(min(5000, len(sample_dataset))):
        record = sample_dataset.iloc[i]
        dataset_plus = calculate_distance(record, dataset)
        dist = dataset_plus.iloc[0]
        print("Record", i, dist, dataset_plus.iloc[1] - dist, dataset_plus.iloc[5] - dist)
        vals.append(dist)
        if (dist < 1):
            valsnth1.append(dataset_plus.iloc[1] - dist)
            valsnth5.append(dataset_plus.iloc[5] - dist)
    logger.write(f"[STAT]\tAverage DCR (mean, median)\t{sum(vals)/len(vals)}\t{statistics.median(vals)}\n")
    logger.write(f"[STAT]\tAverage DCR Delta 1 (mean, median)\t{sum(valsnth1)/len(valsnth1)}\t{statistics.median(valsnth1)}\n")
    logger.write(f"[STAT]\tAverage DCR Delta 5 (mean, median)\t{sum(valsnth5)/len(valsnth5)}\t{statistics.median(valsnth5)}\n")
    vals = torch.tensor(vals, dtype=torch.float)
    valsnth1 = torch.tensor(valsnth1, dtype=torch.float)
    valsnth5 = torch.tensor(valsnth5, dtype=torch.float)
    percentiles = [0.01, 0.03, 0.05, 0.1, 0.2, 0.25, 0.5]
    perc = torch.tensor([0.01, 0.03, 0.05, 0.1, 0.2, 0.25, 0.5], dtype=torch.float)
    actual_percentiles = list(torch.quantile(vals, perc, dim=0).numpy())
    stats = '\n'.join([f"[SUBINFO]\t{x}\t{y}" for x,y in zip(percentiles, actual_percentiles)])
    logger.write(f"\n[SUBINFO]\tQuantiles\t{stats}\n")
    actual_percentiles = list(torch.quantile(valsnth1, perc, dim=0).numpy())
    stats = '\n'.join([f"[SUBINFO]\t{x}\t{y}" for x,y in zip(percentiles, actual_percentiles)])
    logger.write(f"\n[SUBINFO]\tQuantiles Delta 1\n{stats}\n")
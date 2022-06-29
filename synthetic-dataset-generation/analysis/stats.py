import sys, traceback

def plot_correlations(js, dataset, sample_dataset, logger, plot_corr, plot_corr_specific):
    from dython.nominal import associations
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    res = associations(sample_dataset, annot=False, plot=False, ax=ax)
    fig.savefig(plot_corr_specific)
    fig, ax = plt.subplots(figsize=(6, 6))
    res2 = associations(dataset, annot=False, plot=False, ax=ax)
    fig.savefig(plot_corr)
    logger.write(f"[STAT]\tAverage Corr Difference:\t{(res['corr'] - res2['corr']).abs().mean().mean()}\n")

def kskl_divergence(js, dataset, sample_dataset, logger, error_logger):
    from scipy.stats import ks_2samp, entropy
    columns = list(dataset.keys())
    logger.write("[INFO]\tKL Divergence\n")
    ksdivergence = []
    kldivergence = []
    for i in range(len(columns)):
        column = columns[i]
        val = None
        try:
            val = 1 - ks_2samp(dataset[column], sample_dataset[column]).statistic
        except:
            try:
                val = 1 - ks_2samp(dataset[column].astype(str), sample_dataset[column].astype(str)).statistic
            except:
                logger.write(f"[ERROR]\tCould not calculate column\t{column}\n")
                traceback.print_exc(file=error_logger)
        if val is not None:
            ksdivergence.append(val)
            logger.write(f"[SUBINFO]\tKS\t{column}\t{val}\n")

        kl = None
        try:
            minlength = min(len(dataset[column]), len(sample_dataset[column]))
            kl = entropy(dataset[column][:minlength], sample_dataset[column][:minlength])
        except TypeError:
            pass # non float lines
        except:
            traceback.print_exc()
            pass
        if (kl is not None): 
            logger.write(f"[SUBINFO]\tKL\t{column}\t{kl}\n")
            if (kl != float('inf')):
                kldivergence.append(kl)
    logger.write(f"[STAT]\tAverage KS-Divergence:\t{sum(ksdivergence)/len(ksdivergence)}\n")
    if len(kldivergence) > 0:
        logger.write(f"[STAT]\tAverage Non-infinity KL-Divergence:\t{sum(kldivergence)/len(kldivergence)}\n")
    else:
        logger.write(f"[STAT]\tAverage Non-infinity KL-Divergence:\tN/A\n")


def table_evaluator(js, dataset, sample_dataset, logger, error_logger):
    from table_evaluator import TableEvaluator
    real = dataset
    fake = sample_dataset
    discrete_columns = js['discrete_columns']
    target_column = js['target_column']
    # in the case of the target column, if fake doesn't generate ANY samples, we need to artificially change one row to allow the
    # evaluator to work
    if len(fake[target_column].unique()) == 1:
        # oh no! artificially edit a value to create the class imbalance
        # Insure every class is accounted for
        for i, val in enumerate(real[target_column].unique()):
            fake[target_column][i] = val
    try:
        table_evaluator = TableEvaluator(real, fake, cat_cols=discrete_columns)
        target_type = "class"
        if (target_column not in discrete_columns):
            target_type = "regr"
        table_eval = table_evaluator.evaluate(target_col=js['target_column'], target_type=target_type, verbose=True)
        logger.write("[INFO]\tTable Evaluator Results:\n")
        for k in table_eval:
            logger.write(f"[STAT]\t{k}:\t{table_eval[k]}\n")
        logger.flush()
    except Exception as e:
        logger.write(f"[ERROR]\t{e}\n")
        traceback.print_exc(file=error_logger)
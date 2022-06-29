from argparse import Namespace
import json
import pickle
import pandas as pd
import generators
from helpers import HelperException

def _load(folder, generator, args, strict=False):
    js = json.load(open(f"datasets/{folder}/info.json", "r"))
    js['folder'] = folder
    if (generator not in generators.__dict__ and strict):
        raise HelperException("Generator does not exist")
    file = js['name'].lower()
    files = [(file, None)]
    if (args.test):
        files = []
        for i in range(1,args.test_splits+1):
            files.append((f"{file}_train{i}", f"{file}_test{i}"))
    js['files'] = files
    return js

def gen(folder, generator, args=Namespace(), additionals={}):
    from torch import cuda
    import torch
    js = _load(folder, generator, args, strict=True)
    js["metadata"] = json.load(open(f"datasets/{folder}/metadata.json", "r"))
    js['batch_size'] = 200
    js['device'] = "cuda" if cuda.is_available() else "cpu"
    js['additionals'] = lambda x: additionals[x] if x in additionals else None
    files = list(map(lambda x:x[0], js['files']))
    for file in files:
        js['file'] = file
        print("Reading file", file)
        dataset = pd.read_csv(f"datasets/{folder}/{file}.csv", index_col=False) # index column should be removed
        dataset = dataset.dropna()
        js['columns'] = list(dataset)

        for count in range(args.count):
            print(f"Generating {folder} with {generator} #", count)
            ggen_count = str(count).zfill(2)
            model = generators.__dict__[generator].train(js, dataset, epochs=args.gen_epochs)
            try:
                torch.save(model, f"models/{folder}/{file}_{generator}{args.postfix}-{ggen_count}.pt")
                if "extra_pickle" not in js:
                    js["extra_pickle"] = {}
                with open(f"models/{folder}/{file}_{generator}{args.postfix}-{ggen_count}.pickle", 'wb') as outfile:
                    pickle.dump(js['extra_pickle'], outfile)
            except:
                print("Could not save model")
            samples = generators.__dict__[generator].generate(js, model, sample_total=args.gen_sample_count)
            samples.to_csv(f"datasets_synthetic/{folder}/{file}_{generator}{args.postfix}-{ggen_count}.csv", index=False)

def genonly(folder, generator, args=Namespace(), additionals={}):
    from torch import cuda
    import torch
    js = _load(folder, generator, args, strict=True)
    js["metadata"] = json.load(open(f"datasets/{folder}/metadata.json", "r"))
    js['batch_size'] = 200
    js['device'] = "cuda" if cuda.is_available() else "cpu"
    js['additionals'] = lambda x: additionals[x] if x in additionals else None
    files = list(map(lambda x:x[0], js['files']))
    for file in files:
        js['file'] = file
        print("Reading file", file)
        for count in range(args.count):
            print(f"Generating {folder} with {generator} #", count)
            ggen_count = str(count).zfill(2)
            try:
                model = torch.load(f"models/{folder}/{file}_{generator}{args.postfix}-{ggen_count}.pt")
                try:
                    js['extra_pickle'] = pickle.load(open(f"models/{folder}/{file}_{generator}{args.postfix}-{ggen_count}.pickle", "rb"))
                except:
                    print("No pickle file")
                samples = generators.__dict__[generator].generate(js, model, sample_total=args.gen_sample_count)
                samples.to_csv(f"datasets_synthetic/{folder}/{file}_{generator}{args.postfix}-{ggen_count}.csv", index=False)
            except:
                print("Could not find model")

def stats(folder, generator, args=Namespace(), additionals={}):
    from analysis.stats import kskl_divergence, plot_correlations, table_evaluator
    js = _load(folder, generator, args)
    files = list(map(lambda x:x[0], js['files']))
    for file in files:
        js['file'] = file
        print("Reading file", file)
        dataset = pd.read_csv(f"datasets/{folder}/{file}.csv", index_col=False) # index column should be removed
        dataset = dataset.dropna()
        plot_corr = f"datasets_analysis/{folder}/{file}_corr.png"

        for count in range(args.count):
            print(f"Stats for {folder} with {generator} #", count)
            ggen_count = str(count).zfill(2)
            sample_dataset = pd.read_csv(f"datasets_synthetic/{folder}/{file}_{generator}{args.postfix}-{ggen_count}.csv", index_col=False)
            sample_dataset = sample_dataset.dropna()
            for col in sample_dataset.columns:
                sample_dataset[col] = sample_dataset[col].astype(dataset[col].dtype)
            plot_corr_specific = f"datasets_analysis/{folder}/{file}_{generator}{args.postfix}_corr-{ggen_count}.png"
            log_path = f"datasets_analysis/{folder}/{file}_{generator}{args.postfix}_stats-{ggen_count}.txt"
            log_path_error = f"datasets_analysis/{folder}/{file}_{generator}{args.postfix}_error-{ggen_count}.stats.log"
            logger = open(log_path, "w")
            error_logger = open(log_path_error, "w")
            kskl_divergence(js, dataset, sample_dataset, logger, error_logger)
            plot_correlations(js, dataset, sample_dataset, logger, plot_corr, plot_corr_specific)
            table_evaluator(js, dataset, sample_dataset, logger, error_logger)
            logger.close()
            error_logger.close()

def privacy(folder, generator, args=Namespace(), additionals={}):
    from analysis.privacy import dcr
    js = _load(folder, generator, args)
    files = list(map(lambda x:x[0], js['files']))

    for file in files:
        js['file'] = file
        print("Reading file", file)
        dataset = pd.read_csv(f"datasets/{folder}/{file}.csv", index_col=False) # index column should be removed
        dataset = dataset.dropna()
        for count in range(args.count):
            print(f"Stats for {folder} with {generator} #", count)
            ggen_count = str(count).zfill(2)
            sample_dataset = pd.read_csv(f"datasets_synthetic/{folder}/{file}_{generator}{args.postfix}-{ggen_count}.csv", index_col=False)
            sample_dataset = sample_dataset.dropna()
            for col in sample_dataset.columns:
                sample_dataset[col] = sample_dataset[col].astype(dataset[col].dtype)
            log_path = f"datasets_analysis/{folder}/{file}_{generator}{args.postfix}_privacy-{ggen_count}.txt"
            log_path_error = f"datasets_analysis/{folder}/{file}_{generator}{args.postfix}_error-{ggen_count}.privacy.log"
            logger = open(log_path, "w")
            error_logger = open(log_path_error, "w")
            dcr(js, dataset, sample_dataset, logger, error_logger)
            logger.close()
            error_logger.close()

def eval(folder, generator, args=Namespace(), additionals={}):
    from analysis.eval import simple_eval
    js = _load(folder, generator, args)
    files = js['files']

    for file, test_file in files:
        js['file'] = file
        print("Reading file", file)
        dataset = pd.read_csv(f"datasets/{folder}/{file}.csv", index_col=False) # index column should be removed
        dataset = dataset.dropna()
        test_dataset = pd.read_csv(f"datasets/{folder}/{test_file}.csv", index_col=False) # index column should be removed
        test_dataset = test_dataset.dropna()
        for count in range(args.count):
            print(f"Eval for {folder} with {generator} #", count)
            ggen_count = str(count).zfill(2)
            sample_dataset = pd.read_csv(f"datasets_synthetic/{folder}/{file}_{generator}{args.postfix}-{ggen_count}.csv", index_col=False)
            sample_dataset = sample_dataset.dropna()
            for col in sample_dataset.columns:
                sample_dataset[col] = sample_dataset[col].astype(dataset[col].dtype)
            log_path = f"datasets_analysis/{folder}/{file}_{generator}{args.postfix}_eval-{ggen_count}.txt"
            log_path_error = f"datasets_analysis/{folder}/{file}_{generator}{args.postfix}_error-{ggen_count}.eval.log"
            logger = open(log_path, "w")
            error_logger = open(log_path_error, "w")
            simple_eval(js, dataset, test_dataset, sample_dataset, logger, error_logger)
            logger.close()
            error_logger.close()


def aggregate(folder, generator, args=Namespace(), additionals={}):
    import glob
    js = _load(folder, generator, args)
    files = list(map(lambda x:x[0], js['files']))
    for file in files:
        txt_files = glob.glob(f"datasets_analysis/{folder}/{file}_{generator}{args.postfix}_*.txt")
        log_path = f"datasets_analysis/{folder}/{file}_{generator}{args.postfix}-SUMMARY.txt"
        logger = open(log_path, "w")
        logger.write(f"{folder}/{file}_{generator}{args.postfix}\n")
        # We want to aggregate all the stats
        dict_ = {}
        for txt_file in txt_files:
            with open(txt_file,"r") as f:
                for line in f.readlines():
                    line = line.strip().split("\t")
                    if (line[0] == "[STAT]"):
                        # Okay let's congregate that information
                        if (line[1] not in dict_):
                            dict_[line[1]] = []
                        vals = line[2:]
                        parsed_vals = []
                        for val in vals:
                            try:
                                val = float(val)
                            except:
                                pass
                            parsed_vals.append(val)
                        dict_[line[1]].append(parsed_vals)
        for key in sorted(dict_.keys()):
            # Multiple statistics
            final_results = []
            for i in range(len(dict_[key][0])):
                try:
                    new_array = list(map(lambda x:x[i], dict_[key])) # TODO out of bounds?
                    if (all([type(x) == int or type(x) == float or len(x) == 0 for x in new_array])):
                        final_results.append(str(sum(new_array)/len(new_array))) # Average this shit!
                    else:
                        final_results.append("N/A")
                except:
                    print("One of the stats is not aligned")
            logger.write(f"{key}\t" + "\t".join([str(x) for x in final_results]) + "\n")
        logger.close()
        print(f"Summarized {generator} and saved to {log_path}")
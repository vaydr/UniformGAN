print("Initializing Upload Script -- may take a while!")

import glob
import os 
import json
import pandas as pd
from helpers import *
from sklearn.model_selection import train_test_split
import traceback

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_file(filename):
    data = json.load(filename)
    return data

def create_metadata_file(js, files):
    fields = {}
    for column in js['all_columns_reference_DO_NOT_EDIT']:
        if column in js['discrete_columns']:
            fields[column] = {"type": "categorical"}
        else:
            fields[column] = {"type": "numerical", "subtype": "float"}
    tables = {}
    for file in files:
        tables[file] ={
                "fields": fields,
                "path": file + ".csv",
                "target_column": js['target_column']
            }
    output = {
        "tables": tables
    }
    return output
    
def save_json_file(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=2)

def load_json_file(filename):
    data = json.load(open(filename, 'r'))
    return data

def upload(test=False, test_split=0.2, test_splits=3, is_playground=False):
    files = glob.glob(f"{dir_path}/upload/*.csv")
    print("Files available (select number): ")
    for line in [f"{i}: " + file.split("/")[-1] for i, file in enumerate(files)]:
        print(line)

    line_num = input("Select line: ")
    print("***************************")
    try:
        line_num = int(line_num)
        filepath = files[line_num]
        filename = filepath.split("/")[-1]
        info_file = f"{filepath}.json"
        playground_file = f"{filepath}.html"
        if (not os.path.isfile(info_file)):
            print("Could not find corresponding JSON file...")
            print("Auto generating file ... done")
            print(f"Edit {info_file} before re-running")
            default_name = filename.split(".")[0].capitalize()
            dataset = pd.read_csv(filepath)
            if (is_playground):
                from pivottablejs import pivot_ui
                pivot_ui(dataset, outfile_path=playground_file)
                print(f"Outputed {playground_file} Playground")
            save_json_file({
                'name': default_name,
                'target_column': '[FILL ME IN]',
                'discrete_columns': [],
                'ordinal_discrete_columns': {},
                'all_columns_reference_DO_NOT_EDIT': list(dataset),
                'index_column': None,
            }, info_file)
        else:
            js = load_json_file(info_file)
            folder = f"{dir_path}/datasets/{js['name']}/"
            filename = js['name'].lower()
            if (is_playground):
                from pivottablejs import pivot_ui
                dataset = pd.read_csv(filepath, index_col=js['index_column'])
                pivot_ui(dataset, outfile_path=playground_file)
                print(f"Outputed {playground_file} Playground")
                exit()
            print("File contents:")
            print(js)
            c = input("Are you sure to continue? [y/N]: ")
            print("***************************")
            if (c.strip().lower() not in ['y', 'yes']):
                print("Decided not to continue")
                exit()
            
            try:
                os.makedirs(folder)
                data = pd.read_csv(filepath, index_col=js['index_column'])
                files = [filename]
                if (test):
                    for i in range(1,test_splits + 1):
                        train, test = train_test_split(data, test_size=test_split)
                        train.to_csv(f'{folder}/{filename}_train{i}.csv', index=False, header=True)
                        test.to_csv(f'{folder}/{filename}_test{i}.csv', index=False, header=True)
                        files.append(f"{filename}_train{i}")
                data.to_csv(f'{folder}/{filename}.csv')
                js['all_columns_reference_DO_NOT_EDIT'] = list(data)
                save_json_file(js, f"{folder}/info.json")
                save_json_file(create_metadata_file(js, files), f"{folder}/metadata.json")
                print("Successfully created dataset files")
                print("***************************")
            except FileExistsError:
                print(f"Folder already exists. Delete datasets/{js['name']} to rerun.")

    except Exception:
        traceback.print_exc()
        print("*******")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload files')
    parser.add_argument('--test', metavar='test', type=str2bool, nargs='?',
                        const=True, default=False, help="Activate test mode.")
    parser.add_argument('--test_split_ratio', metavar='test_split_ratio', type=float, default=0.2,
                        help='Split mode')
    parser.add_argument('--test-splits', metavar='test_splits', type=int, default=2,help="test splits")
    parser.add_argument('--verbose', metavar='verbose', type=str2bool, nargs='?',
                    const=True, default=False, help="verbosity for logging")
    parser.add_argument('--playground', metavar='playground', type=str2bool, nargs='?',
                    const=True, default=False, help="html for table exploration")
                    

    args = parser.parse_args()
    if (args.verbose):
        print("Args:", args)
    print("***************************")
    if (not args.test):
        print("YOU ARE IN PRODUCTION MODE~! Use --test if you are creating this for testing purposes")
    upload(test=args.test, test_split=args.test_split_ratio, test_splits=args.test_splits, is_playground=args.playground)
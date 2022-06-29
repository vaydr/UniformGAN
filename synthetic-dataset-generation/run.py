from helpers import * 
from pathlib import Path
import actions
import traceback

def run(args):
    arg_actions = list(filter(None, args.actions.split(",")))
    arg_generators = list(filter(None, args.generators.split(",")))
    arg_datasets =  list(filter(None, args.datasets.split(",")))
    if ("gen" in arg_actions and len(arg_actions) > 1):
        print("Cannot generate new data and also apply other actions")
        return
    if (len(arg_actions) == 0 or len(arg_generators) == 0):
        print("No action and/or generators selected")
        return

    arg_additionals = {}
    for line in args.additionals.split(","):
        line = line.strip()
        if len(line) == 0:
            continue
        spl = line.split(":")
        if (len(spl) > 1):
            arg_additionals[spl[0]] = spl[1]

    if (args.verbose):
        arg_additionals["verbose"] = True

    for folder in arg_datasets:
        Path(f"datasets_synthetic/{folder}").mkdir(parents=True, exist_ok=True)
        Path(f"datasets_analysis/{folder}").mkdir(parents=True, exist_ok=True)
        Path(f"models/{folder}").mkdir(parents=True, exist_ok=True)
        for action in arg_actions:
            if (action in actions.__dict__):
                for generator in arg_generators:
                    try:
                        actions.__dict__[action](folder, generator, args=args, additionals=arg_additionals)
                    except HelperException as e:
                        print("An error occured with", action, generator, e)
                        print("Continuing....")
                    except Exception as e:
                        print("An error occured with", action, generator, e)
                        if (args.verbose):
                            traceback.print_exc()
                        print("Continuing....")
            else:
                print(action, " could not be executed because it is not a valid action")
    
if __name__ == "__main__":
    print("Synthetic Dataset Generation")
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--config', metavar='config', help="Config file (relative path)", type=str)
    parser.add_argument('--verbose', metavar='verbose', type=str2bool, nargs='?',
                        const=True, default=False, help="verbosity for logging")
    parser.add_argument('--test', metavar='test', type=str2bool, nargs='?',
                        const=True, default=False, help="Activate test mode.")
    parser.add_argument('--test-splits', metavar='test_splits', type=int, default=2,help="test splits ")
    parser.add_argument('--datasets', metavar='folder', type=str, default="Adult", help="Datasets (i.e. Adult)")
    parser.add_argument('--generators', metavar='generators', type=str, default="ctgan,tablegan,uniformgan", help="Generators (i.e. ctgan)")
    parser.add_argument('--actions', metavar='actions', type=str, default="", help="Actions (i.e. gen,stats,privacy,eval)")
    parser.add_argument('--count', metavar='count', type=int, default=3, help="Number of copies of the dataset (default 3)")
    parser.add_argument('--postfix', metavar='postfix', type=str, default="",help="postfix name tag for synthetic dataset (i.e. _v1)")
    parser.add_argument('--gen-epochs', metavar='gen_epochs', type=int, default=200,help="epochs")
    parser.add_argument('--gen-sample-count', metavar='gen_sample_count', type=int, default=40000,help="number of samples")
    parser.add_argument("--additionals", metavar='additionals', type=str, default="", help="A string with additional attributes. E.g. key:value,key2:value2")
    # TODO implement gen_additional
    args = parser.parse_args()
    if (args.config):
        with open(args.config, "r") as f:
            data = parser.parse_args(f.read().split(), namespace=None)
            for k, v in vars(data).items():
                if getattr(args, k, None) is None:
                    setattr(args, k, v)
    print(args)
    if (not args.test):
        print("YOU ARE IN PRODUCTION MODE~! Use --test if you are creating this for testing purposes")
    run(args)
    print("DONE")
import argparse

# KILL ALL
# LLstat | grep temp | cut -d " " -f1 | sed 's/.*/LLkill &/'

import os
def run_script(dataset, action, generator, others):
    with open("temp.sh", "w") as f:
        if (action != "privacy" and action != "eval"):
            f.write("""#!/bin/bash
# 10 cores (memory)
#SBATCH -c 6
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/{0}.log-%j

# Loading the required module
source /etc/profile
module load anaconda/2021a
source env/bin/activate

# Run the script""".format(action))
        else:           
            f.write("""#!/bin/bash
# 10 cores (memory)
#SBATCH -c 4
#SBATCH -o logs/{0}.log-%j

# Loading the required module
source /etc/profile
module load anaconda/2021a
source env/bin/activate

# Run the script""".format(action))
        f.write(f"\npython run.py --datasets={dataset} --actions={action} --generators={generator} {others}\n")
        f.close()
    os.system("LLsub temp.sh")
    os.remove("temp.sh")

if __name__ == "__main__":
    print("Synthetic Dataset Generation Supercloud addition")
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--actions', metavar='actions', type=str, default="", help="Actions (i.e. gen,stats,privacy,eval)")
    parser.add_argument('--datasets', metavar='folder', type=str, default="Adult", help="Datasets (i.e. Adult)")
    parser.add_argument('--generators', metavar='generators', type=str, default="ctgan,tablegan,uniformgan", help="Generators (i.e. ctgan)")
    args, unknown_args = parser.parse_known_args()
    for dataset in args.datasets.split(","):
        if len(dataset) == 0:
            continue
        for generator in args.generators.split(","):
            if len(generator) == 0:
                continue
            for action in args.actions.split(","):
                if len(action) == 0:
                    continue
                others = " ".join(unknown_args)
                print(f"Running {action} on {generator} with {others}")
                run_script(dataset, action, generator, others)

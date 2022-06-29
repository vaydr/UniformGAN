def complete():
    import glob
    txt_files = glob.glob(f"datasets_analysis/*/*-SUMMARY.txt")
    logger = open("SUMMARY.txt", "w")
    dict_ = {} # attribute -> { [generator] : value/values }
    generators = set()
    longest_stat = 1
    for file in txt_files:
        lines = open(file, "r").readlines()
        # LINE 1 is always the name
        generator_name, lines = lines[0].strip(), lines[1:]
        generators.add(generator_name)
        # We want to aggregate all the stats
        for line in lines:
            line = [l.strip() for l in line.strip().split("\t")]
            # [stat name] \t [stat values...]
            stat_name, stats = line[0], line[1:]
            print(stats)
            longest_stat = max(len(stats), longest_stat)
            if (stat_name not in dict_):
                dict_[stat_name] = {}
            dict_[stat_name][generator_name] = stats
    generators = list(generators)    
    column_names=["Statistic"]
    for generator in sorted(generators):
        column_names.append(generator)
        for i in range(1, longest_stat):
            column_names.append("")
    logger.write("\t".join(column_names) + "\n")
    for stat_name in sorted(dict_.keys()):
        row = [stat_name]
        for i in range(1, len(column_names)):
            column = column_names[i]
            if len(column) == 0 or column not in dict_[stat_name]:
                if len(row) <= i:
                    row.append("")
            else:
                row.extend(dict_[stat_name][column])
        logger.write("\t".join(row) + "\n")
    logger.close()

if __name__ == "__main__":
    complete()
        

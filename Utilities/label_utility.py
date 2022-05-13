import pandas as pd

# Get all labels from the dataset which always co-occur with some other label
# For example, the label @nuxt/devalue always occur together with the label @nuxtjs/devalue
# Input:
# DATASET_PATH = path to the file containing the whole dataset
# OUTPUT_PATH = path to the output file where the list of co-occured labels are written
def find_cooccur_labels(DATASET_PATH, OUTPUT_PATH):
    description_fields = ["cve_id", "cleaned"]
    df = pd.read_csv(DATASET_PATH, usecols=description_fields)
    # Read column names from file
    cols = list(pd.read_csv(DATASET_PATH, nrows=1))
    # print(cols)
    pd_labels = pd.read_csv(DATASET_PATH,
                            usecols=[i for i in cols if i not in ["cve_id", "cleaned", "matchers", "merged"]])

    # Get the list of available labels
    list_labels = [item for item in cols if item not in ["cve_id", "cleaned", "matchers", "merged"]]

    # Save to a string array too
    list_same_columns = []
    # Loop through all labels, find those that are duplicate
    for i in range(len(list_labels)):
        for j in range(i + 1, len(list_labels)):
            if pd_labels[list_labels[i]].equals(pd_labels[list_labels[j]]):
                list_same_columns.append(list_labels[i] + " == " + list_labels[j])
                print(list_labels[i] + " == " + list_labels[j])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        for line in list_same_columns:
            file.write(line + "\n")

# Combine labels that always co-occur (i.e., a label that always occur together with other label)
# Utilize the file produced by find_cooccur_labels function
# Input:
# DATASET_PATH = path to the file containing the whole dataset
# COOCCUR_DATA_PATH = path to the output file from the find_cooccur_labels function
# OUTPUT_PATH = path to the output file which will contain the whole dataset with the co-occuring labels merged as one string
def combine_labels(DATASET_PATH, COOCCUR_DATA_PATH, OUTPUT_PATH):
    # read the csv
    df = pd.read_csv(DATASET_PATH)
    # Build the label dictionary
    label_dict = {}
    with open(COOCCUR_DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().lstrip()
            labels = line.split(" == ")
            if labels[0] in label_dict:
                label_dict[labels[0]].append(labels[1])
            else:
                label_dict[labels[0]] = [labels[1]]
        f.close()
    # Build the name change dictionary (i.e., change between original name and the combined merged name)
    name_change_dictionary = {}
    for key, arr in label_dict.items():
        # Check if the right hand side (i.e., the other label) exist in the column
        for value in arr:
            if value in df.columns:
                # print("INSIDE")
                # If exist, merge the other label name to the original label name
                if key in name_change_dictionary:
                    name_change_dictionary[key] = name_change_dictionary[key] + ";" + value
                else:
                    name_change_dictionary[key] = key + ";" + value
                # Then, drop the other label name
                df.drop(value, axis=1, inplace=True)
    # Then, change the column names based on name change dictionary
    df.rename(columns=name_change_dictionary, inplace=True)

    # Write to file
    df.to_csv(OUTPUT_PATH)
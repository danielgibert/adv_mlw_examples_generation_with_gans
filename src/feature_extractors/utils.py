
def save_all_strings(allstrings: list, output_filepath: str):
    with open(output_filepath, "w") as output_file:
        for string in allstrings:
            output_file.write("{}\n".format(string))

def load_all_strings(input_filepath: str):
    all_strings = []
    with open(input_filepath, "r") as input_file:
        for line in input_file.readlines():
            all_strings.append(line.strip())
    return all_strings
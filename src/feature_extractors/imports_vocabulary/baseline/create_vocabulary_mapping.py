import json
from collections import OrderedDict

with open("small_dll_imports.json", "r") as input_file:
    data = json.load(input_file)

vocabulary_mapping = OrderedDict()
inverse_vocabulary_mapping = OrderedDict()

i = 0
for lib in data:
    for function in data[lib]:
        vocabulary_mapping["{};{}".format(lib.lower(), function)] = i
        inverse_vocabulary_mapping[i] = "{};{}".format(lib.lower(), function)
        i += 1

with open("vocabulary/vocabulary_mapping.json", "w") as output_file:
    json.dump(vocabulary_mapping, output_file)

with open("vocabulary/inverse_vocabulary_mapping.json", "w") as output_file:
    json.dump(inverse_vocabulary_mapping, output_file)
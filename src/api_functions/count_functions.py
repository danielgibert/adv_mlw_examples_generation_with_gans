import argparse
import os
from src.pe_modifier import PEModifier
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate boxplot')
    parser.add_argument("executables_filepath",
                        type=str,
                        help="Directory where the executables features are stored in binary format")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Output filepath")
    args = parser.parse_args()

    API_libraries = {}

    filenames = os.listdir(args.executables_filepath)
    total_api_libraries = 0
    total_api_functions = 0
    for i, filename in enumerate(filenames):
        print("{};{}".format(i, filename))
        try:
            pe_modifier = PEModifier(os.path.join(args.executables_filepath, filename))
            if pe_modifier.lief_binary is None:
                print("Exception loading binary!")
            else:
                for lib in pe_modifier.lief_binary.imports:
                    print(lib.name)
                    if lib.name.lower() not in API_libraries.keys():
                        API_libraries[lib.name.lower()] = {}
                        total_api_libraries += 1
                    for entry in lib.entries:
                        if not entry.is_ordinal:
                            try:
                                API_libraries[lib.name.lower()][entry.name] += 1
                            except KeyError as ke:
                                API_libraries[lib.name.lower()][entry.name] = 1
                                total_api_functions += 1
                print("#API libraries: {}; #functions: {}".format(total_api_libraries, total_api_functions))
        except Exception as e:
            print(e)

    print(API_libraries)
    with open(args.output_filepath, "w") as output_file:
        json.dump(API_libraries, output_file)







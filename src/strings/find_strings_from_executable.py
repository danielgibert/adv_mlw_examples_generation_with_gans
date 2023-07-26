import argparse
from src.pe_modifier import PEModifier
import os
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find strings from executable')
    parser.add_argument("executable_filepath",
                        type=str,
                        help="Filepath of the executable")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Output file containing the strings found")
    args = parser.parse_args()

    allstrings = re.compile(b'[\x20-\x7f]{5,}')

    #filenames = os.listdir(args.executables_filepath)
    total_strings = 0
    encoding = 'utf-8'

    print("{}".format(args.executable_filepath))
    try:
        pe_modifier = PEModifier(args.executable_filepath)
        if pe_modifier.lief_binary is None:
            print("Exception loading binary!")
        else:
            executable_strings = allstrings.findall(pe_modifier.bytez)

            with open(args.output_filepath, "w") as output_file:
                for string in executable_strings:
                    print(string)
                    output_file.write("{}\n".format(str(string, encoding)))
            print(executable_strings)
    except Exception as e:
        print(e)




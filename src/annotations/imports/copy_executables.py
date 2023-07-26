import argparse
import sys
sys.path.append("../../../")
import os
from src.pe_modifier import PEModifier
from shutil import copyfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy executables from one directory to another')
    parser.add_argument("executables_filepath",
                        type=str,
                        help="Input executables filepath")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Output executables filepath")
    args = parser.parse_args()

    filenames = os.listdir(args.executables_filepath)
    for i, filename in enumerate(filenames):
        try:
            pe_modifier = PEModifier(os.path.join(args.executables_filepath, filename))
            # Copy executable
            copyfile(os.path.join(args.executables_filepath, filename),
                     os.path.join(args.output_filepath, pe_modifier.sha256))
        except Exception as e:
            print(e)
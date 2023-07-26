import argparse
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select top K API functions')
    parser.add_argument("api_functions_filepath",
                        type=str,
                        help="CSV-like file containing the counts for each API function invoked in the dataset")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Output file"
                        )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of API functions to retrieve",
        default=150
    )
    args = parser.parse_args()

    i = 1
    with open(args.output_filepath, "w") as output_file:
        output_file.write("key,value\n")
        with open(args.api_functions_filepath, "r") as input_file:
            reader = csv.DictReader(input_file, fieldnames=["key", "value"])
            reader.__next__()
            for row in reader:
                tokens = row["key"].split(";")
                try:
                    if tokens[1] != '':
                        print(tokens)
                        output_file.write("{},{}\n".format(row["key"], row["value"]))
                        if i == args.k:
                            break
                        i += 1

                except IndexError as ie:
                    print(ie)



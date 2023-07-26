import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate boxplot')
    parser.add_argument("functions_filepath",
                        type=str,
                        help="JSON-like file containing the counts for each API function invoked in the dataset")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Output file"
                        )
    args = parser.parse_args()

    API_libraries_v2 = {}
    with open(args.functions_filepath, "r") as functions_file:
        API_libraries = json.load(functions_file)
        print(API_libraries.keys())

    for api in API_libraries.keys():
        for function in API_libraries[api]:
            API_libraries_v2["{};{}".format(api,function)] = API_libraries[api][function]
    i = 0

    with open(args.output_filepath, "w") as output_file:
        output_file.write("key,value\n")
        for k, v in sorted(API_libraries_v2.items(), key=lambda item: item[1], reverse=True):
            output_file.write("{},{}\n".format(k,v))
            print(i, k, v)
            i+=1

import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate barplot')
    parser.add_argument("functions_filepath",
                        type=str,
                        help="JSON-like file containing the functions invoked")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Output file")
    args = parser.parse_args()

    library_count = {}

    with open(args.functions_filepath, "r") as functions_file:
        API_libraries = json.load(functions_file)
        print(API_libraries.keys())

    for api in API_libraries.keys():
        for function in API_libraries[api]:
            try:
                library_count[api] += API_libraries[api][function]
            except KeyError as ke:
                library_count[api] = API_libraries[api][function]
    i = 0
    with open(args.output_filepath, "w") as output_file:
        output_file.write("key,value\n")
        for k, v in sorted(library_count.items(), key=lambda item: item[1], reverse=True):
            output_file.write("{},{}\n".format(k,v))
            print(i, k, v)
            i+=1

import json


ember_features_filepaths = ["/filepath_to_EMBER_2017_v2/train_features_0.jsonl",
                            "/filepath_to_EMBER_2017_v2/train_features_1.jsonl",
                            "/filepath_to_EMBER_2017_v2/train_features_2.jsonl",
                            "/filepath_to_EMBER_2017_v2/train_features_3.jsonl",
                            "/filepath_to_EMBER_2017_v2/train_features_4.jsonl",
                            "/filepath_to_EMBER_2017_v2/train_features_5.jsonl"]

api_libraries = {}
i = 0
for ember_features_filepath in ember_features_filepaths:
    with open(ember_features_filepath, "r") as jsonl_file:
        json_list = list(jsonl_file)
    for json_str in json_list:
        result = json.loads(json_str)
        if result["label"] == 0:
            print("{}".format(i))

            for api_library in result["imports"]:
                if api_library.lower() not in api_libraries.keys():
                    api_libraries[api_library.lower()] = {}
                for api_function in result["imports"][api_library]:
                    try:
                        api_libraries[api_library.lower()][api_function] += 1
                    except KeyError as ke:
                        api_libraries[api_library.lower()][api_function] = 1
        i += 1

with open("data/ember_2017/ember_api_functions.json", "w") as output_file:
    json.dump(api_libraries, output_file)

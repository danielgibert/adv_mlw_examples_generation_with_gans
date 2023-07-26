import numpy as np
#import lief
from collections import OrderedDict
import hashlib
#import argparse
#import json
#import sys
#sys.path.append("../../")
#from adversarial_malware_samples_generation.pe_modifier import PEModifier
from adversarial_malware_samples_generation.feature_extractors.feature_type import FeatureType
#from adversarial_malware_samples_generation.feature_extractors.hashed_imports_info_extractor import HashedImportsInfoExtractor
from sklearn.feature_extraction import FeatureHasher


class ImportsInfoExtractor(FeatureType):
    ''' Information about imported libraries and functions from the
    import address table.  Note that the total number of imported
    functions is contained in GeneralFileInfo.
    '''

    name="imports"
    dim = 18797 #That is a lot of features

    def __init__(self, vocabulary_mapping, inverse_vocabulary_mapping):
        super(FeatureType, self).__init__()
        self.vocabulary_mapping = vocabulary_mapping
        self.inverse_vocabulary_mapping = inverse_vocabulary_mapping
        self.feature_labels = [feature_label for feature_label in self.vocabulary_mapping.keys()]
        self.feature_vector = OrderedDict({feature_label: 0 for feature_label in self.vocabulary_mapping.keys()})
        self.dim = len(self.vocabulary_mapping.keys())

    def raw_features(self, bytez, lief_binary):
        total_counts = 0
        imports = {}
        if lief_binary is None:
            return imports

        for lib in lief_binary.imports:
            if lib.name.lower() not in imports:
                imports[lib.name.lower()] = []  # libraries can be duplicated in listing, extend instead of overwrite

            # Clipping assumes there are diminishing returns on the discriminatory power of imported functions
            #  beyond the first 10000 characters, and this will help limit the dataset size
            for entry in lib.entries:
                if entry.is_ordinal:
                    imports[lib.name.lower()].append("ordinal" + str(entry.ordinal))
                else:
                    imports[lib.name.lower()].append(entry.name[:10000])
                    try:
                        self.feature_vector["{};{}".format(lib.name.lower(), entry.name)] = 1
                        total_counts += 1
                    except KeyError as ke:
                        print("{};{}".format(lib.name.lower(), entry.name))
        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features["hashed_imports"] = imports
        features[self.name] = self.feature_vector
        return features

    def process_raw_features(self, raw_obj):
        features = [raw_obj[self.name][k] for k in raw_obj[self.name].keys()]
        return np.array(features, dtype=np.float32)

    def get_imports_features_from_ember_json(self, ember_raw_obj):
        feature_vector = OrderedDict({feature_label: 0 for feature_label in self.vocabulary_mapping.keys()})
        for library_name in ember_raw_obj["imports"].keys():
            for function_name in ember_raw_obj["imports"][library_name]:
                try:
                    feature_vector["{};{}".format(library_name.lower(), function_name)] = 1
                except KeyError as e:
                    continue
        features = [feature_vector[k] for k in feature_vector.keys()]
        return features



    def update_imports_dictionary(self, imports_dict, original_features_array, fake_features_array):
        bitwise_xor_arr = np.bitwise_xor(original_features_array, fake_features_array)
        if bitwise_xor_arr.shape[0] == 1:
            indices = np.where(bitwise_xor_arr[0] == 1)[0]
        else:
            indices = np.where(bitwise_xor_arr == 1)[0]
        for i in indices:
            flabel = self.feature_labels[i].split(";")
            lib = flabel[0]
            entry = flabel[1]
            try:
                imports_dict[lib].append(entry)
            except KeyError as ke:
                imports_dict[lib] = []
                imports_dict[lib].append(entry)
        return imports_dict

    def create_imports_dictionary(self, original_features_array, fake_features_array):
        imports_dict = {}
        maximum_arr = np.maximum(original_features_array, fake_features_array)
        indices = np.where(maximum_arr == 1)[0]
        for i in indices:
            flabel = self.feature_labels[i].split(";")
            lib = flabel[0]
            entry = flabel[1]
            try:
                imports_dict[lib].append(entry)
            except KeyError as ke:
                imports_dict[lib] = []
                imports_dict[lib].append(entry)
        return imports_dict


    def apply_hashing_trick(self, imports_dict):
        libraries = list(set([l.lower() for l in imports_dict.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]

        # A string like "kernel32.dll:CreateFileMappingA" for each imported function
        imports = [lib.lower() + ':' + e for lib, elist in imports_dict.items() for e in elist]
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]

        # Two separate elements: libraries (alone) and fully-qualified names of imported functions
        # Now we should have the same feature vector as the EMBER imports features
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract API import functions given an executable')
    parser.add_argument("executable_filepath",
                        type=str,
                        help="Executable filepath")
    parser.add_argument("vocabulary_mapping",
                        type=str,
                        help="Vocabulary mapping lib;fun -> index")
    parser.add_argument("inverse_vocabulary_mapping",
                        type=str,
                        help="Vocabulary mapping  index -> lib;fun")
    args = parser.parse_args()

    with open(args.vocabulary_mapping, "r") as input_file:
        vocabulary_mapping = json.load(input_file)

    with open(args.inverse_vocabulary_mapping, "r") as input_file:
        inverse_vocabulary_mapping = json.load(input_file)

    pe_modifier = PEModifier(args.executable_filepath)
    feature_extractor = ImportsInfoExtractor(vocabulary_mapping, inverse_vocabulary_mapping)

    raw_obj = feature_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
    feature_array = feature_extractor.process_raw_features(raw_obj)
    processed_features_v1 = feature_extractor.apply_hashing_trick(raw_obj["hashed_imports"], feature_array)

    hashed_feature_extractor = HashedImportsInfoExtractor()
    hashed_features = hashed_feature_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
    processed_features_v2 = hashed_feature_extractor.process_raw_features(hashed_features)

    print(processed_features_v1)
    print(processed_features_v2)

    for i in range(len(processed_features_v1)):
        if processed_features_v1[i] != processed_features_v2[i]:
            raise Exception()
"""
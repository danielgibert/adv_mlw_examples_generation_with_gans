import sys
import hashlib
import numpy as np
import argparse
sys.path.append("../../")
from adversarial_malware_samples_generation.feature_extractors.feature_type import FeatureType
from adversarial_malware_samples_generation.pe_modifier import PEModifier


class ByteHistogramExtractor(FeatureType):
    ''' Byte histogram (count + non-normalized) over the entire binary file '''

    name = 'histogram'
    dim = 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        """
        Generates a JSON-able representation of the file containing its SHA256 value and the bytes histogram
        :param bytez:
        :param lief_binary:
        :return:
        """
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features[self.name] = counts.tolist()
        return features

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj[self.name], dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract API import functions given an executable')
    parser.add_argument("executable_filepath",
                        type=str,
                        help="Executable filepath")
    args = parser.parse_args()

    pe_modifier = PEModifier(args.executable_filepath)
    feature_extractor = ByteHistogramExtractor()

    raw_obj = feature_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
    feature_array = feature_extractor.process_raw_features(raw_obj)
    print(feature_array)



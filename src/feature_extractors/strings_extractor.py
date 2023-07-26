from adversarial_malware_samples_generation.feature_extractors.feature_type import FeatureType
import numpy as np
import re
from collections import OrderedDict
import hashlib


class StringsExtractor(FeatureType):
    name = 'strings'
    dim = 20000

    def __init__(self, vocabulary_mapping, inverse_vocabulary_mapping):
        super(FeatureType, self).__init__()
        self.vocabulary_mapping = vocabulary_mapping
        self.inverse_vocabulary_mapping = inverse_vocabulary_mapping
        self.feature_labels = [feature_label for feature_label in self.vocabulary_mapping.keys()]
        self.feature_vector = OrderedDict({feature_label: 0 for feature_label in self.vocabulary_mapping.keys()})
        self.dim = len(self.vocabulary_mapping.keys())
        self.encoding = 'utf-8'


        # all consecutive runs of 0x20 - 0x7f that are 5+ characters
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        self.allstrings = []
        # occurances of the string 'C:\'.  Not actually extracting the path
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # occurances of http:// or https://.  Not actually extracting the URLs
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        # occurances of the string prefix HKEY_.  No actually extracting registry names
        self._registry = re.compile(b'HKEY_')
        # crude evidence of an MZ header (dropper?) somewhere in the byte stream
        self._mz = re.compile(b'MZ')

    def _reset(self):
        self.feature_vector = OrderedDict({feature_label: 0 for feature_label in self.vocabulary_mapping.keys()})

    def raw_features(self, bytez, lief_binary):
        self._reset()
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # statistics about strings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0

        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        strings_statistics_dict = {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printabledist': c.tolist(),  # store non-normalized histogram
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }
        features["strings_statistics"] = strings_statistics_dict

        for string in allstrings:
            string = str(string, self.encoding)
            self.allstrings.append(string)
            try:
                self.feature_vector[string] = 1
            except KeyError as e:
                pass
        features[self.name] = self.feature_vector
        return features

    def process_raw_features(self, raw_obj):
        """
        1 if feature appears. Otherwise, 0
        """
        features = [raw_obj[self.name][k] for k in raw_obj[self.name].keys()]
        return np.array(features, dtype=np.float32)

    def process_statistic_features(self, raw_obj):
        """
        The same as process_raw_features() in StringsStatisticsExtractor
        """
        hist_divisor = float(raw_obj["strings_statistics"]['printables']) if raw_obj["strings_statistics"]['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj["strings_statistics"]['numstrings'],
            raw_obj["strings_statistics"]['avlength'],
            raw_obj["strings_statistics"]['printables'],
            np.asarray(raw_obj["strings_statistics"]['printabledist']) / hist_divisor,
            raw_obj["strings_statistics"]['entropy'],
            raw_obj["strings_statistics"]['paths'],
            raw_obj["strings_statistics"]['urls'],
            raw_obj["strings_statistics"]['registry'],
            raw_obj["strings_statistics"]['MZ']
        ]).astype(np.float32)

    def get_allstrings(self):
        return self.allstrings

    def update_hashed_features(self, original_features: np.ndarray, fake_features: np.ndarray, all_strings: list, hashed_strings_features: np.ndarray):
        bitwise_xor_arr = np.bitwise_xor(original_features, fake_features)
        indices = np.where(bitwise_xor_arr == 1)[0]
        for i in indices:
            all_strings.append(self.inverse_vocabulary_mapping[str(i)])

        all_strings_as_bytes = [str.encode(string) for string in all_strings]
        if all_strings_as_bytes:
            # statistics about strings:
            string_lengths = [len(s) for s in all_strings_as_bytes]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(all_strings_as_bytes)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0

        strings_statistics_dict = {
            'numstrings': len(all_strings_as_bytes),
            'avlength': avlength,
            'printabledist': c.tolist(),  # store non-normalized histogram
            'printables': int(csum),
            'entropy': float(H),
            'paths': hashed_strings_features[-4],
            'urls': hashed_strings_features[-3],
            'registry': hashed_strings_features[-2],
            'MZ': hashed_strings_features[-1]
        }

        hist_divisor = float(strings_statistics_dict['printables']) if strings_statistics_dict['printables'] > 0 else 1.0
        updated_hashed_strings_features = np.hstack([
            strings_statistics_dict['numstrings'],
            strings_statistics_dict['avlength'],
            strings_statistics_dict['printables'],
            np.asarray(strings_statistics_dict['printabledist']) / hist_divisor,
            strings_statistics_dict['entropy'],
            strings_statistics_dict['paths'],
            strings_statistics_dict['urls'],
            strings_statistics_dict['registry'],
            strings_statistics_dict['MZ']
        ]).astype(np.float32)
        return updated_hashed_strings_features

    def retrieve_strings_to_inject(self, strings_features: np.ndarray, fake_strings_features: np.ndarray, inverse_vocabulary_mapping: dict):
        strings_to_inject = []
        print(strings_features.shape, fake_strings_features.shape)
        bitwise_xor_arr = np.bitwise_xor(strings_features, fake_strings_features)
        if bitwise_xor_arr.shape[0] == 1:
            indices = np.where(bitwise_xor_arr[0] == 1)[0]
        else:
            indices = np.where(bitwise_xor_arr == 1)[0]

        for i in indices:
            strings_to_inject.append(inverse_vocabulary_mapping[str(i)])
        return strings_to_inject


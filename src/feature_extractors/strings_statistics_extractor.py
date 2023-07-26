from adversarial_malware_samples_generation.feature_extractors.feature_type import FeatureType
import numpy as np
import re
import hashlib


class StringsStatisticsExtractor(FeatureType):
    ''' Extracts strings from raw byte stream '''
    name = 'strings'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1

    def __init__(self):
        super(FeatureType, self).__init__()
        # all consecutive runs of 0x20 - 0x7f that are 5+ characters
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        # occurances of the string 'C:\'.  Not actually extracting the path
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # occurances of http:// or https://.  Not actually extracting the URLs
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        # occurances of the string prefix HKEY_.  No actually extracting registry names
        self._registry = re.compile(b'HKEY_')
        # crude evidence of an MZ header (dropper?) somewhere in the byte stream
        self._mz = re.compile(b'MZ')

    def raw_features(self, bytez, lief_binary):
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
        features[self.name] = {
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
        return features

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj[self.name]['printables']) if raw_obj[self.name]['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj[self.name]['numstrings'], raw_obj[self.name]['avlength'], raw_obj[self.name]['printables'],
            np.asarray(raw_obj[self.name]['printabledist']) / hist_divisor, raw_obj[self.name]['entropy'], raw_obj[self.name]['paths'], raw_obj[self.name]['urls'],
            raw_obj[self.name]['registry'], raw_obj[self.name]['MZ']
        ]).astype(np.float32)
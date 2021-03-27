import re
import itertools
import pandas as pd
import math
from urllib.parse import urlparse


""" -------------------------------------------------------
This URL parser implements the approach taken in the paper 
"Fast webpage classification using URL features" written by 
Min-Yen Kan and Hoang Oanh Nguyen Thi, as well as "Web page 
categorization without the web page" by Min-Yen Kan.
-------------------------------------------------------"""


unigram_frequencies = dict(pd.read_csv('data/unigram_freq.csv', index_col=False).values)
TOTAL_COUNT = sum(unigram_frequencies.values())


def segment_by_baseline(url_string):
    """
        Segment a URL into its components as given by the URI protocol,
        and then further break these components at non-alphanumeric
        characters and URI-escaped entities (eg. '%20').
    """
    parse_result = urlparse(url_string)
    components = {
        'scheme': parse_result.scheme,
        'netloc': parse_result.netloc,
        'path': parse_result.path,
        'params': parse_result.params,
        'query': parse_result.query,
        'fragment': parse_result.fragment
    }
    DELIMITERS = r'[^a-zA-Z\d\s]'
    for component in components:
        components[component] = list(filter(len, re.split(DELIMITERS, components[component])))
    return components


def segment_by_information_content(components):
    """
        Segment tokens containing concatenated words
        into its component words.
    """

    def get_info_content(tkn):
        """
            We make use of the following corpus to calculate the
            Information Content (IC) of each partition.
            Source: https://www.kaggle.com/rtatman/english-word-frequency
        """
        freq = 0 if tkn not in unigram_frequencies else unigram_frequencies[tkn]
        return 1000000 if freq == 0 else -math.log(freq / TOTAL_COUNT)

    def get_best_partition(tkn):
        """
            This function implements the recursive strategy of partitioning
            proposed by Min-Yen Kan, which does not guarantee the global
            minimum entropy partitioning but matches the local minima in most
            cases. It has a much lower time complexity of O(n log n).
        """
        if len(tkn) == 1:
            return [tkn]
        tkn_entropy = get_info_content(tkn)
        splits = [(tkn[:i], tkn[i:]) for i in range(1, len(tkn))]
        entropies = [sum([get_info_content(frag) for frag in split]) for split in splits]
        if min(entropies) < tkn_entropy:
            best_split = splits[entropies.index(min(entropies))]
            best_partition = [get_best_partition(frag) for frag in best_split]
            return [it for lst in best_partition for it in lst]
        return [tkn]

    for component in components:
        best_partitions = [get_best_partition(token) for token in components[component]]
        components[component] = [it for lst in best_partitions for it in lst]
    return components


def parse(url_string):
    baseline_segmented_components = segment_by_baseline(url_string)
    return segment_by_information_content(baseline_segmented_components)


test_url1 = "http://audience.cnn.com/services/activatealert.jsp" + \
           "?source=cnn&id=203&value=hurricane+isabel"
test_url3 = "http://www.christianmusicdaily.com"
print(parse(test_url3))

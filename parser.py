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
        Segment tokens containing concatenated words into its component
        words.
    """

    def get_all_partitions(tkn):
        """
            Returns a list of all possible partitions of a token.
            Source: https://stackoverflow.com/questions/37023774/
                    all-ways-to-partition-a-string
        """
        parts = []
        for cut_points in range(1 << (len(tkn) - 1)):
            result = []
            last_cut = 0
            for i in range(len(tkn) - 1):
                if (1 << i) & cut_points != 0:
                    result.append(tkn[last_cut:(i + 1)])
                    last_cut = i + 1
            result.append(tkn[last_cut:])
            parts.append(result)
        return parts

    def get_info_content(part):
        """
            We make use of the following corpus to calculate the
            Information Content (IC) of each partition.
            Source: https://www.kaggle.com/rtatman/english-word-frequency
        """
        ic_sum = 0
        for frag in part:
            freq = 0 if frag not in unigram_frequencies else unigram_frequencies[frag]
            ic = 1000000 if freq == 0 else -math.log(freq / TOTAL_COUNT)
            ic_sum += ic
        return ic_sum

    def get_best_partition(tkn):
        parts = get_all_partitions(tkn)
        chars = list(tkn)
        ics = [get_info_content(partition) for partition in parts]
        chars_ic = get_info_content(chars)
        best_part = parts[ics.index(min(ics))] if min(ics) < chars_ic else [tkn]
        return best_part

    for component in components:
        best_partitions = [get_best_partition(token) for token in components[component]]
        components[component] = list(itertools.chain.from_iterable(best_partitions))
    return components


def parse(url_string):
    baseline_segmented_components = segment_by_baseline(url_string)
    return segment_by_information_content(baseline_segmented_components)


test_url1 = "http://audience.cnn.com/services/activatealert.jsp" + \
           "?source=cnn&id=203&value=hurricane+isabel"
test_url2 = "http://cs.cornell.edu/Info/Courses/Current/CS415/CS414.html"
test_url3 = "http://audience.cnn.com/services/naturallanguageprocessing.jsp" + \
           "?source=cnn&id=203&value=hurricane+isabel"
print(parse(test_url1))

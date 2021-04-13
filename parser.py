import re
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


def segment_by_component(url_string):
    """
        Segment a URL into its components as given by the URI protocol.
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
    return components


def segment_by_baseline(components):
    """
        Further break these components at non-alphanumeric
        characters and URI-escaped entities (eg. '%20').
    """
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
        return 10000 * len(tkn) if freq == 0 else -math.log(freq / TOTAL_COUNT)

    def get_best_partition(tkn):
        """
            This function improves on the recursive strategy of partitioning
            proposed by Min-Yen Kan. By assigning a much greater entropy value
            for tokens of greater length, it is able to give accurate partitions
            for more than 2 concatenated words.
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
    return [it for lst in
            list(segment_by_information_content(
                segment_by_baseline(
                    segment_by_component(url_string.lower()))).values())
            for it in lst]


# Parse balanced data and save as CSV for future usage
train = pd.read_csv('data/balanced_data_3210.csv', header=None)
raw_X_data = train[0].tolist()
X_parsed = pd.Series([" ".join(parse(url)) for url in raw_X_data])
y_labels = pd.Series(train[1].tolist())
parsed_df = pd.concat([X_parsed, y_labels], axis=1)
parsed_df.to_csv('balanced_parsed_data_3210.csv', header=False, index=False)

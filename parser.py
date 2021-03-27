import re
import itertools
from urllib.parse import urlparse


""" -------------------------------------------------------
This URL parser implements the approach taken in the paper 
"Fast webpage classification using URL features" written by 
Min-Yen Kan and Hoang Oanh Nguyen Thi.
-------------------------------------------------------"""


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
        words. To do so, we make use of the UMBC WebBase Corpus to
        calculate the Information Content (IC) of each partition.
        Source: https://ebiquity.umbc.edu/resource/html/id/351
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

    def get_info_content(partition):
        return 1

    for component in components:
        for token in components[component]:
            partitions = get_all_partitions(token)
            characters = list(token)
            info_contents = [get_info_content(partition) for partition in partitions]
            chars_info_content = get_info_content(characters)
            if min(info_contents) < chars_info_content:
                best_partition = partitions[info_contents.index(min(info_contents))]
                token = best_partition
        components[component] = list(itertools.chain.from_iterable(components[component]))
    return components


def parse(url_string):
    baseline_segmented_components = segment_by_baseline(url_string)
    return baseline_segmented_components


test_url1 = "http://audience.cnn.com/services/activatealert.jsp" + \
           "?source=cnn&id=203&value=hurricane+isabel"
test_url2 = "http://cs.cornell.edu/Info/Courses/Current/CS415/CS414.html"
print(parse(test_url2))

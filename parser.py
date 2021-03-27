import re
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

    DELIMITERS = r'[^a-zA-Z\d\s]'
    split_url = list(filter(len, re.split(DELIMITERS, url_string)))
    return split_url


def parse(url_string):
    return segment_by_baseline(url_string)


test_url = "http://audience.cnn.com/services/activatealert.jsp" + \
           "?source=cnn&id=203&value=hurricane+isabel"
print(parse(test_url))

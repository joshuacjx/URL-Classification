# from urllib.parse import urlparse
import re


class Parser:

    def parse(self, url_string):
        DELIMITERS = r'[:/.-]'
        split_url = re.split(DELIMITERS, url_string)
        return list(filter(lambda token: token != '', split_url))


parser = Parser()
print(parser.parse("http://www.cm-life.com/news/2003/09/12/features/librarians.also.help.on.web-2490840.shtml"))

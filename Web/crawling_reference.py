import csv
from lxml import html
import requests
from urllib.parse import urlparse
import time
import pandas as pd
import numpy as np
import ast
from bs4 import BeautifulSoup
import re
import time


def crawl_bugs_launchpad(URL):
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    try:
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        print(soup)
        description = soup.find_all("span", class_="yui3-editable_text-text ellipsis")[0].text.strip()
        description = _RE_COMBINE_WHITESPACE.sub(" ", description)
    except Exception as e: 
        print(e)
        description = ""
    return description


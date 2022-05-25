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


_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
def crawl_bugs_launchpad(URL):
    try:
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        description = soup.find_all("span", class_="yui3-editable_text-text ellipsis")[0].text.strip()
        description = _RE_COMBINE_WHITESPACE.sub(" ", description)
    except Exception as e: 
        description = ""
    return description


def crawl_openwall(URL):
    try:
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        description = soup.find_all("pre")[0].text.split("============")[0]
        description = _RE_COMBINE_WHITESPACE.sub(" ", description).split("Affected Versions")[0]
    except Exception as e: 
        description = ""
    return description


def crawl_bugzilla_redhat(URL):
    try:
        re = requests.get(URL)
        soup = BeautifulSoup(re.text, "html.parser")
        description = str(soup.find_all("div", {"class": "uneditable_textarea"}))
        
        # perform additional text processing to clean scraped features
        if description != "":
            description = description[34:]
            description = description[:description.find("<")]

        if description == '':
            mini_soup = soup.find_all("div", {"id": "c0"})
            if len(mini_soup) > 0:
                comment = str(mini_soup[0].find_all("pre", {"class": "bz_comment_text"}))
                test = str(comment).splitlines()
                store = ""
                for x in range(0, len(test)):
                    item = test[x].lower()
                    if "description" in item:
                        for line in test[x+1:]:
                            if line != "":
                                if line[len(line)-1] != ":":
                                    store += line + " "
                                else:
                                    break

                    if "summary" in item:
                        for line in test[x+1:]:
                            if line != "":
                                if line[len(line)-1] != ":":
                                    store += line + " "
                                else:
                                    break
                        break

                description = store
    except Exception as e: 
        description = ""
    return description


def crawl_access_redhat(CVE,URL):
    try:
        re = requests.get(URL)
        soup = BeautifulSoup(re.text, "html.parser")
        
        #obtained scraped features by extracting associated elements: (description, affected_products)
        description = soup.find_all("div", {"id": "description"})
        affected_products = soup.find_all("div", {"id": "affected_products"})
        
        test = str(affected_products).split("<li>")
        itemlist = []
        for item in test:
            if "</li>" in item:
                item = item[0:item.find("</li>")].strip()
                if ">" in item:
                    start = item.find(">") + 1
                    end = item[item.find(">"):].find("<")
                    item = item[start:start+end-1]
                itemlist.append(item)
        affected_products = itemlist
        
        listed = str(description).split("<li>")
        for item in listed:
            if CVE in item:
                final = item[item[1:].find(" ")+2:item.find(CVE)-2]
        description = final
    except Exception as e: 
        description = ""
    return description



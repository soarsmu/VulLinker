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

def crawl_rhn_redhat(URL):
    try:
        re = requests.get(URL)
        soup = BeautifulSoup(re.text, "html.parser")
        
        #obtained scraped features by extracting associated elements: (description, affected_products)
        descriptions = soup.find_all("div", {"id": "description"})
        description = ""
        for x in descriptions:
            for y in x.find_all('p'):
                description += y.text +" "
    except Exception as e: 
        description = ""
    return description


def crawl_lists_debian(URL):
    try:
        re = requests.get(URL)
        soup = BeautifulSoup(re.text, "html.parser")
        soup = str(soup)
        
        test = soup[soup.find("<pre>")+5:soup.find("</pre>")]
        test = test.splitlines()
        
        # Affected products logic variables
        affp_go_1 = False
        affp_go_2 = False
        affp = ""
        
        # CVE Description logic variables
        description_go = False
        description_go_count = 0
        description = ""
        
        for x in test:
            
            # Affected products logic
            if affp_go_2:
                if x != "":
                    affp += x + " "
                if x == "" and affp != "":
                    affp_go_1 = False
                    affp_go_2 = False
            if affp_go_1:
                if x == "":
                    affp_go_2 = True
            if "debian bug" in x.lower():
                affp_go_1 = True
                
            # CVE description logic
            if description_go:
                if x != "":
                    description += x 
                else:
                    description_go_count += 1
                    if description_go_count == 2:
                        description_go = False
                    
            if y[0] == x:
                description_go = True
                
        description = description.strip()
        description = description.replace("  ", " ")
        description = description.replace("  ", " ")
    except Exception as e: 
        description = ""
    return description


def crawl_debian(URL):
    try:
        re = requests.get(URL)
        soup = BeautifulSoup(re.text, "html.parser")  
        body = str(soup.find_all("div", {"id": "content"}))  
        new_body = body[body.find("More information"):]
        
        description = new_body[new_body.find("<p>")+3: new_body.find("</p>")]
    except Exception as e: 
        description = ""
    return description


def crawl_oracle(URL):
    try:
        re = requests.get(URL)
        soup = BeautifulSoup(re.text, "html.parser")  
        test = str(soup).splitlines()
        
        go = False
        aff_p = []
        product = ""
        counter = 0
        for x in test:
            if counter == 2:
                go = False
                counter = 0
                aff_p.append(product)
                product = ""
            if go and counter < 2:
                product += x[4:len(x)-5] + " "
                counter += 1
            if ">CVE-2019-0190<" in x:
                go = True
                
        description = ""
        for header in soup.find_all('h3', string="Description"):
            nextNode = header
            while True:
                nextNode = nextNode.nextSibling
                if nextNode is None:
                    break
                if nextNode.name is not None:
                    if nextNode.name == "h3":
                        break
                    description += nextNode.get_text(strip=True)
    except Exception as e: 
        description = ""
    return description


def crawl_lists_opensuse(CVE,URL):
    try:
        re = requests.get(URL)
        soup = BeautifulSoup(re.text, "html.parser")  
        body = soup.find_all("div", {"class": "email-body"})
        test = str(body).splitlines()

        go = False
        go2 = False
        go3 = False
        aff_p = []
        fin_aff_p = []
        desc = ""
        for x in test:
            x = str(x.strip())
            if len(x) != 0:
                if "______________________________________________________________________________" in x:
                    go = False
                    fin_aff_p = aff_p 

                if go:
                    aff_p.append(x)

                if "Affected Products" in x:
                    go = True

                if "Description:" in x:
                    go2 = True 

                if go2:
                    if ("-" == x[0] or "*" == x[0]) and CVE in x:
                        go3 = True

                    if (("-" == x[0] or "*" == x[0]) and CVE not in x) or "Patch Instructions:" in x:
                        go3 = False
                        description = ""
                        description += desc[desc.find(":")+2:]

                if go3:
                    desc += x + " "
    except Exception as e: 
        description = ""
    return description
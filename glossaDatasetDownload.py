"""

Lucía's script to download XML files of articles from Glossa journal.

"""

import os
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import json
import xml.etree.ElementTree as ET

# Constants
BASE = "https://www.glossa-journal.org"
ARTICLES_PAGE = BASE + "/articles"
PATH_CORPUS = 'input_data/glossa_corpus/'

# Fetch and parse HTML
def get_soup(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

# Extract article links from a listing page
def get_article_links(page_url):
    soup = get_soup(page_url)
    links = []
    # links to article pages typically contain "/articles/" in href
    for a in soup.select("a"):
        href = a.get("href") or ""
        if "/article/" in href and href.count("/") > 2:
            # normalize to full URL
            full = href if href.startswith("http") else BASE + href
            links.append(full)
    return list(set(links))

# Extract XML download link from an article page
def get_xml_link(article_url):
    soup = get_soup(article_url)
    xml_link = None
    for a in soup.find_all("a"):
        text = (a.text or "").strip().lower()
        if "xml" in text:
            href = a.get("href")
            if href:
                xml_link = href if href.startswith("http") else BASE + href
                break
    return xml_link

# Download a page and save HTML
def download_page(url, output_path):
    resp = requests.get(url)
    resp.raise_for_status()
    filename = resp.headers['Content-Disposition'].split("filename=")[-1].strip('"')
    path_to_filename = output_path + filename
    with open(path_to_filename, "w", encoding="utf-8") as f:
            f.write(resp.text)

# Extract metadata from a page
def get_meta(soup, name):
    tag = soup.find("meta", attrs={"name": name})
    return tag["content"] if tag else None

def main():

    # Main loop to gather links
    all_xml_links = []
    all_article_links = []

    for page in range(1, 50):
        print(f"Fetching list page {page}")
        url = f"{ARTICLES_PAGE}?page={page}"
        try:
            article_urls = get_article_links(url)
        except Exception as e:
            print("Stop paging:", e)
            break

        if not article_urls:
            break

        for art in article_urls:
            all_article_links.append(art)
            try:
                xml = get_xml_link(art)
                if xml:
                    all_xml_links.append(xml)
            except Exception as ex:
                print("    Failed to fetch XML:", ex)
            time.sleep(0.3)
        time.sleep(1)
        print(f'Page {page} done')

    print("Removing duplicates and downloading XML files...")
    # Remove duplicates
    all_article_links = list(set(all_article_links))
    all_xml_links = list(set(all_xml_links))
    links = {'article_links': all_article_links, 'xml_links': all_xml_links}

    # Download XML files
    with tqdm(total=1026) as pbar:
        for xml in links['xml_links']:
            download_page(xml, PATH_CORPUS)
            pbar.update(1)

    print("Total articles:", len(links['article_links']))
    print("Total XML files:", len(links['xml_links']))
          
    print("Creating JSON with articles information...")
    # Create a json with all the articles information
    articles = {}

    for link in links['article_links']:
        id = link.split('/')[-2]
        article = {'id': link.split('/')[-2], 'article_link': link}
        articles[id] = article

    for link in links['xml_links']:
        xml_id = link.split('/')[4]
        if xml_id  in articles.keys():
            articles[link.split('/')[4]]['download_xml_link'] = link
    
    with tqdm(total=1026) as pbar:
        for article in articles:
            url = articles[article]['article_link']
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            metadata = {}
            metadata['title'] = get_meta(soup, "DC.Title")
            metadata['abstract'] = get_meta(soup, "DC.Description")
            if get_meta(soup, "citation_keywords"):
                metadata['keywords'] = get_meta(soup, "citation_keywords").split(";")
            else: 
                metadata['keywords'] = []
            metadata['doi'] = get_meta(soup, "DC.Identifier.DOI")
            metadata['journal'] = get_meta(soup, "DC.Source")
            metadata['volume'] = get_meta(soup, "DC.Source.Volume")
            metadata['issue'] = get_meta(soup, "DC.Source.Issue")
            metadata['date_issued'] = get_meta(soup, "DC.Date.issued")
            metadata['date_modified'] = get_meta(soup, "DC.Date.modified")
            metadata['language'] = get_meta(soup, "DC.Language")
            metadata['pdf_url'] = get_meta(soup, "citation_pdf_url")
            metadata['xml_url'] = get_meta(soup, "citation_xml_url")
            articles[article]['metadata'] = metadata
            pbar.update(1)

    # Extract authors from XML files and add to JSON
    for folder in os.listdir("input_data/glossa_corpus/"):
        folder_path = os.path.join("input_data/glossa_corpus/", folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                for article in articles:
                    if file.endswith(".xml") and file.__contains__(article):
                        f = open(folder_path + "/" + file, "r", encoding="utf-8")
                        xml_data = f.read()
                        xml_data = xml_data.replace("&nbsp;", " ")
                        root_text = ET.fromstring(xml_data)
                        authors = []
                        for contrib in root_text.findall(".//contrib[@contrib-type='author']"):
                            name_tag = contrib.find("name")
                            if name_tag is not None:
                                given = name_tag.findtext("given-names", default="")
                                surname = name_tag.findtext("surname", default="")
                                full_name = f"{given} {surname}".strip()
                                authors.append(full_name)
                        articles[article]['metadata']['authors'] = authors
                        f.close()

    with open(PATH_CORPUS + 'glossa_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
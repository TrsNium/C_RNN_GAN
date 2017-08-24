from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import urlretrieve
import re
import os

content_url = "https://freemidi.org/"
html_doc = urlopen(content_url+"genre").read()
sp = BeautifulSoup(html_doc)

genres = sp.find_all("div", {"class":"genre-big-ones"})
genres_href = [[tag["href"] for tag in genre.find_all("a")] for genre in genres]

genre_artist = {}
selected_genre = ["genre-hip-hop-rap", "genre-disco"]
for hrefs in genres_href:
    for href in hrefs:
        if not href in selected_genre:
            continue
        print(href)
        content = {}
        url = content_url + href
        html_doc = BeautifulSoup(urlopen(url).read())
        artist_hrefs = html_doc.find_all("div", {"class": "genre-link-text"})
        for artist_href in artist_hrefs:
            artist_href_ = artist_href.a.get("href")
            a_html_doc = BeautifulSoup(urlopen(content_url+artist_href_).read())
            content[artist_href.a.string] = {re.sub("\r\n\s{2,}", "", artist_href.a.string):a.get("href") for a in a_html_doc.find_all("a", {"itemprop":"url"})[1:]}
        genre_artist[href] = content

for key, item in genre_artist.items():
    if not os.path.exists(key):
        os.mkdir(key)
        
    for artist_n, song_dict in item.items():
        if not os.path.exists(key+"/"+artist_n):
            os.mkdir(key+"/"+artist_n)
            
        for song_n, song_href in song_dict.items():
            try:
                href = BeautifulSoup(urlopen(content_url+song_href).read()).find("a", {"id":"downloadmidi"}).get("href")
                urlretrieve(content_url+href, key+"/"+artist_n+"/"+song_n+".mid")
            except:
                continue
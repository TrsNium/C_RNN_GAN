from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import urlretrieve
import re
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

content_url = "https://freemidi.org/"
html_doc = urlopen(content_url+"genre").read()
sp = BeautifulSoup(html_doc)

genres = sp.find_all("div", {"class":"genre-big-ones"})
genres_href = [[tag["href"] for tag in genre.find_all("a")] for genre in genres]

'''
[['genre-rock', 'genre-pop', 'genre-hip-hop-rap', 'genre-rnb-soul'],
 ['genre-classical', 'genre-country', 'genre-jazz', 'genre-blues'],
 ['genre-dance-eletric', 'genre-folk', 'genre-punk', 'genre-newage'],
 ['genre-reggae-ska', 'genre-metal', 'genre-disco', 'genre-bluegrass']]
'''

genre_artist = {}
selected_genre = ["genre-hip-hop-rap", "genre-dance-eletric", "genre-disco"]
for hrefs in genres_href:
    for href in hrefs:
        if not href in selected_genre:
            continue
        content = {}
        url = content_url + href
        html_doc = BeautifulSoup(urlopen(url).read(), "lxml")
        artist_hrefs = html_doc.find_all("div", {"class": "genre-link-text"})
        print(href)
        for artist_href in artist_hrefs:
            artist_href_ = artist_href.a.get("href")
            a_html_doc = BeautifulSoup(urlopen(content_url+artist_href_).read())
            content[artist_href.a.string] = {re.sub("\r\n\s{2,}", "", artist_href.a.string):a.get("href") for a in a_html_doc.find_all("a", {"itemprop":"url"})[1:]}
        genre_artist[href] = content

cwd = os.getcwd()
for key, item in genre_artist.items():
    if not os.path.exists(key):
        os.mkdir(key)
        
    chromeOptions = webdriver.ChromeOptions()
    prefs = {"download.default_directory" : cwd+"/"+key+"/"}
    chromeOptions.add_experimental_option("prefs",prefs)
    
    for artist_n, song_dict in item.items():                           
        for song_n, song_href in song_dict.items():
            try:
                browser = webdriver.Chrome(executable_path="chromedriver", chrome_options=chromeOptions)
                browser.get(content_url+song_href)
                browser.find_element_by_link_text('Download MIDI').click()
                time.sleep(5)
                browser.quit()
            except:
                continue
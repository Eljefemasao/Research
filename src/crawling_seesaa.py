
from bs4 import BeautifulSoup as soup
import requests
from fake_useragent import UserAgent
import time
from selenium import webdriver

BASEURL = "https://www.instagram.com/"

SCOPEPAGE = 'https://www.instagram.com/explore/tags/seesaa/?hl=en'


def fetch(url):
    driver = webdriver.PhantomJS(executable_path='')
    driver.get(url)
    html = driver.page_source

    UA = UserAgent()
    header = {'user-agent': UA.safari}
    r = requests.get(url, headers=header)
    r.encoding = r.apparent_encoding
    return html


def scrape(html):
    img_list = []
    doc = soup(html, 'html.parser')
    print(doc)
    divs = doc.find_all("div", class_="KL4Bh")
    for i in divs:
        img = i.find_all("img", class_='FFVAD')
        img_list.append(img['src'])
    return img_list


def main():

    html = fetch(SCOPEPAGE)
    print(html)
    img_list = scrape(html)

    for i in img_list:
        print(i)


if __name__ == "__main__":
    main()
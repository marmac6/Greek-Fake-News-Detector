from bs4 import BeautifulSoup
import requests
import csv


def getURLS():
    url_list = []
    num_pages = 11
    for i in range(1, num_pages + 1):
        print("page: " + str(i))
        page_url = "https://www.ellinikahoaxes.gr/category/kathgories/scam/page/" + str(
            i)
        result_pages = requests.get(page_url)
        html_pages = BeautifulSoup(result_pages.text, "html.parser")
        urls = html_pages.find_all("div", {"class": "blog-post-content"})
        for u in urls:
            atags = u.find_all('a')
            url_list.append(atags[0]['href'])
    return url_list


def getData(html_article):
    result_article = requests.get(html_article)
    page = BeautifulSoup(result_article.text, "html.parser")
    claim = page.find("div", {"class": "Claim"})
    if claim is not None:
        return claim.p.get_text()
    else:
        return ""


def write_csv(list_fake):

    try:
        with open(file_name, 'w', newline='', encoding='utf-8') as f:
            my_writer = csv.writer(f)
            my_writer.writerow(['title', 'is_fake'])  #add header

            for ind in list_fake:
                if ind is not None:
                    my_writer.writerow([ind, 1])
                else:
                    continue
    except FileNotFoundError as e:
        print(f'Translation aborded. Error: {e}')


file_name = 'myScam.csv'

fake_url_list = getURLS()
fake_news_list = []
for u in fake_url_list:
    fake_news_list.append(getData(u))
fake_news_list_no_null = list(filter(
    None, fake_news_list))  #filter out the null values in the list
write_csv(fake_news_list_no_null)

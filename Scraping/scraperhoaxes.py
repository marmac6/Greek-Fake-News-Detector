from  __future__ import annotations

import csv
import requests
from bs4 import BeautifulSoup


OUTPUT_PATH = 'myScam.csv'


def getURLS(category: str = 'scam') -> list[str]:
    """Retrieves all the urls in the greek hoaxes site and the different
    categories."""
    url_list: list[str] = []
    num_pages = 11
    for i in range(1, num_pages + 1):
        print("page: " + str(i))
        page_url = f"https://www.ellinikahoaxes.gr/category/kathgories/{category}/page/{i}"
        result_pages = requests.get(page_url)
        html_pages = BeautifulSoup(result_pages.text, "html.parser")
        urls = html_pages.find_all("div", {"class": "blog-post-content"})
        for u in urls:
            atags = u.find_all('a')
            url_list.append(atags[0]['href'])
    return url_list


def get_claim_text(html_article: str) -> str:
    """Retrieves the text inside the claim section of the given url."""
    result_article = requests.get(html_article)
    page = BeautifulSoup(result_article.text, "html.parser")
    claim = page.find("div", {"class": "Claim"})
    if claim is not None:
        return claim.p.get_text().strip()


def write_csv(fake_news_entries: list[str]) -> None:
    try:
        with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
            my_writer = csv.writer(f)
            my_writer.writerow(['title', 'is_fake'])  #add header

            for fake_news_entry in fake_news_entries:
                my_writer.writerow([fake_news_entry, 1])
    except FileNotFoundError as e:
        print(f'Scrapping aborted. Error: {e}')


if __name__ == '__main__':
    fake_url_list = getURLS()

    fake_news_list = []
    for url in fake_url_list:
        claim_text = get_claim_text(url)
        if claim_text is not None:
            fake_news_list.append(claim_text)

    write_csv(fake_news_list)

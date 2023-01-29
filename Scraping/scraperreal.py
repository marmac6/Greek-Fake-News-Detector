from __future__ import annotations

import csv
import requests
from bs4 import BeautifulSoup


OUTPUT_PATH = 'myCulture.csv'


def get_titles() -> list[str]:
    titles_list = []
    num_pages = 12
    for i in range(1, num_pages + 1, 5):
        print("page: " + str(i))
        page_url = f"https://www.kathimerini.gr/culture/page/{i}"
        result_pages = requests.get(page_url)
        html_pages = BeautifulSoup(result_pages.text, "html.parser")
        titles = html_pages.find_all("div",
                                     {"class": "design_one_title_medium"})
        for u in titles:
            atags = u.find_all('a')
            titles_list.append(atags[0].string.strip())
    return titles_list


def write_csv(real_news_entries: list[str]) -> None:
    try:
        with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
            my_writer = csv.writer(f)
            my_writer.writerow(['title', 'is_fake'])  #add header

            for real_news_entry in real_news_entries:
                if real_news_entry is not None:
                    my_writer.writerow([real_news_entry, 0])
    except FileNotFoundError as e:
        print(f'Scrapping aborded. Error: {e}')


real_titles = get_titles()
write_csv(real_titles)

from bs4 import BeautifulSoup
import requests
import csv


def getTitles():
    titles_list = []
    num_pages = 100
    for i in range(1, num_pages + 1, 5):
        print("page: " + str(i))
        page_url = "https://www.kathimerini.gr/culture/page" + str(i)
        result_pages = requests.get(page_url)
        html_pages = BeautifulSoup(result_pages.text, "html.parser")
        titles = html_pages.find_all("div",
                                     {"class": "design_one_title_medium"})
        for u in titles:
            atags = u.find_all('a')
            titles_list.append(atags[0].string)
    return titles_list


def write_csv(list_real):

    try:
        with open(file_name, 'w', newline='', encoding='utf-8') as f:
            my_writer = csv.writer(f)
            my_writer.writerow(['title', 'is_fake'])  #add header

            for ind in list_real:
                if ind is not None:
                    my_writer.writerow([ind, 0])
                else:
                    continue
    except FileNotFoundError as e:
        print(f'Translation aborded. Error: {e}')


file_name = 'myCulture.csv'

real_titles = getTitles()
write_csv(real_titles)

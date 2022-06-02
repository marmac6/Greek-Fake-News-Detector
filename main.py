"""
Created on Mon May  9 12:24:21 2022

@author: marmac
"""
import csv
import deepl

auth_key = 'e9c08ffa-9fc9-3eb8-eb0c-eed20d006334:fx'
translator = deepl.Translator(auth_key)

#mia random allagh
def count_chars(file):
    with open(file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)

        char_counter = 0
        for i,row in enumerate(csv_reader):
            statement = row[0]
            statement = _compress_statement(statement)
            char_counter += len(statement)
            
    return char_counter


def _translate_to_greek(statement: str, mock=True) -> str:  #to mock einai orisma
    if mock:
        return statement.replace('a', '2')
    else:
        return translator.translate_text(statement, target_lang="el").text


def _compress_statement(statement: str):
    statement = statement.replace('and', '&')
    statement = statement.replace('dollars', '$')
    statement = statement.replace('percent', '%')
    statement = statement.replace('. ', '.')
    return statement


def translate_statement(statement):
    statement = _compress_statement(statement)
    # call deepl API
    translated = _translate_to_greek(statement, mock=False)
    # replace '&' -> 'και'
    translated = translated.replace('&', 'και')
    return translated


def read_csv(file):
    with open(file, 'r') as read_obj:
        csv_dict_reader = csv.reader(read_obj)

        entries = []
        for row in csv_dict_reader:
            entries.append(row)
        return entries[1:]


def _get_current_index(output_filename):
    try:
        with open(output_filename, 'r') as f:
            return len(f.readlines())
    except FileNotFoundError:
        return 0


def translate_dataset(entries):
    output_filename = f'{output_folder}/translated_{file_name}'
    current_index = _get_current_index(output_filename)
    volume = len(entries)
    

    try:
        with open(output_filename, 'a', newline='', encoding='utf-8') as f:
            mywriter = csv.writer(f)
            if not current_index:
                mywriter.writerow(['Statement', 'Label'])  # add header

            for ind, entry in enumerate(entries):
                if ind < current_index - 1:
                    continue
                print(f'{ind} / {volume}')   #gia na einai akribias den thelei ena +1?
                translated = translate_statement(entry[0])
                mywriter.writerow([translated, entry[1]])
    except FileNotFoundError as e:
        print(f'Translation aborded. Error: {e}')


file_name = 'train.csv'
output_folder = 'translated_with_api'
csv_entries = read_csv(file_name)
translate_dataset(csv_entries)
# c = count_chars(file_name)
# print(c)

import json
import pandas as pd

from model_config import *

def load_data():
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as train_file:
        train_data = json.load(train_file)
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)
    with open(LAW_DATA_PATH, 'r', encoding='utf-8') as law_file:
        law_data = json.load(law_file)
        
    return train_data, test_data, law_data


def matching_data(question_data, law_data):
    new_question_data = []
    for item in question_data:
        id = item['id']
        href = item['href']
        question = item['question']
        answer = item['answer']
        relevant_laws = item['relevant_laws']
        content_Law = ""

        for re_law in relevant_laws:
            name = re_law['name']
            href = re_law['href']
            id_Law = re_law['id_Law']
            id_Chapter = re_law['id_Chapter']
            id_Section = re_law['id_Section']
            id_Article = re_law['id_Article']        

            filtered_law_items = [i for i in law_data if i.get("id") == id_Law]
            if len(filtered_law_items) == 0:
                print(id)
                print(id_Law)
            list_Chapters = filtered_law_items[0]["content"]
            title = filtered_law_items[0]["title"]
            if id_Chapter != "None" or id_Section != "None" or id_Article != "None":
                filered_law_chapter = [i for i in list_Chapters if i.get("id_Chapter") == id_Chapter] if id_Chapter != "None" else [list_Chapters[0]]
                if len(filered_law_chapter) == 0:
                    print(id)
                    print(id_Law)
                list_Sections = filered_law_chapter[0]["content_Chapter"]
                title_Chapter = filered_law_chapter[0]["title_Chapter"]
                filered_law_Section = [i for i in list_Sections if i.get("id_Section") == id_Section] if id_Section != "None" else [list_Sections[0]]
                list_Articles = filered_law_Section[0]["content_Section"]
                title_Section = filered_law_Section[0]["title_Section"]
                filered_law_Article = [i for i in list_Articles if i.get("id_Article") == id_Article] if id_Article != "None" else [list_Articles[0]]

                title_Article = filered_law_Article[0]["title_Article"]
                content_Article = filered_law_Article[0]["content_Article"]

                content_Law = (content_Law + " </s> </s> " + title_Article + " " + content_Article) if content_Law != "" else (content_Law + title_Article + " " + title_Section + " " + title_Chapter + " " + title + " " + content_Article)
        
        if id_Chapter != "None" or id_Section != "None" or id_Article != "None":
            new_question_data.append({
                "id": id,
                "question": question,
                "answer": answer,
                "id_Law": id_Law,
                "id_Chapter": id_Chapter,
                "id_Section": id_Section,
                "id_Article": id_Article,  
                "content_Law": content_Law
            })
            
    return new_question_data


def preprocess_data(data):
    df = pd.DataFrame(data)
    columns_to_remove = ['id_Law', 'id_Chapter', 'id_Section', 'id_Article']
    df = df.drop(columns=columns_to_remove)
    return df


def get_vlqa_data():
    train_data, test_data, law_data = load_data()
    train_data_matched = matching_data(train_data, law_data)
    test_data_matched = matching_data(test_data, law_data)
    df_train = preprocess_data(train_data_matched)
    df_test = preprocess_data(test_data_matched)
    return df_train, df_test
        

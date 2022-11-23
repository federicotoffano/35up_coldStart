import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from pickle import load
import numpy as np
from tensorflow import keras
import re

USER_INPUT = False
LANGS = ['de', 'it', 'pt']
GENDERS = ['F', 'M']
PROFFS = ['programmer', 'self-employed', 'tradesman']
GENRES = ['action', 'adventure',
          'animation', 'comedy', 'crime', 'documentary', 'drama', 'family',
          'fantasy', 'foreign', 'history', 'horror', 'music', 'mystery',
          'romance', 'science fiction', 'thriller', 'tv movie', 'war', 'western']

BINS_PATH = ["model/user_age_binarizer.pkl", "model/user_nationality_binarizer.pkl",
              "model/user_gender_binarizer.pkl", "model/user_occupation_binarizer.pkl",
              "model/genres_binarizer.pkl"]

#load binarizers
bins_list = []
for bin_path in BINS_PATH:
    with open(bin_path, "rb") as input_file:
        user_age_binarizer = load(input_file)
        bins_list.append(user_age_binarizer)

#load classifier
model = keras.models.load_model('model/seq.keras')

#load datasets
df_items_formatted = pd.read_csv('db/items_formatted.csv')
df_users_formatted = pd.read_csv('db/users_formatted.csv')
df_users = pd.read_csv('db/users.csv')
df_items = pd.read_csv('db/items.csv')
df_gens = pd.read_csv('db/genres.csv')

genre_dict = dict(zip(df_gens['id'], df_gens['genre']))

def print_recommendation(user_data):

    age = user_data[0][0]
    if age < 18:
        age_rng = 'Under18'
    elif age < 25:
        age_rng = '18-24'
    elif age < 35:
        age_rng = '25-34'
    else:
        age_rng = '35+'
    user_data[0][0] = age_rng
    
    #build user vector using binarizers
    user_vector = np.array([])
    for index, val in enumerate(user_data):
        vec = bins_list[index].transform([val])[0]
        # print(vec)
        user_vector = np.concatenate((user_vector, vec))
    # print(user_vector)
    
    #input classifier - built user-item list for each item
    user_item_list = np.array([np.array(np.concatenate((user_vector, np.array(row[1:]).tolist())))
                                                for index, row in df_items_formatted.iterrows()])
    #compute item socres
    item_scores = model.predict(user_item_list).flatten()
    # print(item_scores)
    # print(max(item_scores))
    
    #select top items
    n_top_items = 5
    ind = np.argpartition(item_scores, -n_top_items)[-n_top_items:]
    top_items = df_items.iloc[ind]
    top_test_scores = item_scores[ind]
    
    #print recommendations
    recc = 0
    genre_dict = dict(zip(df_gens['id'], df_gens['genre']))
    for index, row in top_items.iterrows():
        gens_indexes = str(row['genres']).split('-')
        #format genres output string
        gen_names = re.compile("(['\]\[])").sub(r'', str([genre_dict[int(gen)] for gen in gens_indexes]))
        original_title = str(row['original_title'])
        original_language = str(row['original_language'])
        year = str(row['year'])
        popularity  = str(row['popularity'])
        print(f'\nRecommendation {recc+1}'
              f'\nTitle: {original_title}'
              f'\nGenres: {gen_names}'
              f'\nOriginal Language: {original_language}'
              f'\nYear: {year}'
              f'\nPopularity: {round(float(popularity),4)}'
              f'\nConfidence of recommendation: {round(float(top_test_scores[recc]),4)}')
        recc+=1



#'id',
# age: '18-24', '25-34', '35+', 'Under18',
# lang: 'de', 'it', 'pt',
# gender: 'F', 'M',
# profession: 'programmer', 'self-employed', 'tradesman',
# genres: 'action', 'adventure',
# 'animation', 'comedy', 'crime', 'documentary', 'drama', 'family',
# 'fantasy', 'foreign', 'history', 'horror', 'music', 'mystery',
# 'romance', 'science fiction', 'thriller', 'tv movie', 'war', 'western'

# input: 'age', 'nationality', 'gender', 'occupation', 'genre1', 'genre2'

if USER_INPUT:
    print('lang: ' + re.compile("(['\]\[])").sub(r'', str(GENDERS)))
    print('lang: ' + re.compile("(['\]\[])").sub(r'', str(PROFFS)))
    print('lang: ' + re.compile("(['\]\[])").sub(r'', str(GENRES)))
    print()
    print('input format: <age> <lang> <gender> <profession> <genre1> <genre2>')
    print('input exaple: 20 de M programmer music western')

    while True:
        try:
            age = int(input('age must be int in range [1,100]\nyour age: '))
            if 0 < age < 101:
                break
        except (ValueError, TypeError):
            pass
    while True:
        lang = input('languages: ' + re.compile("(['\]\[])").sub(r'', str(LANGS)) + '\nyour lang: ')
        if lang in LANGS:
            break
    while True:
        gender = input('genders: ' + re.compile("(['\]\[])").sub(r'', str(GENDERS)) + '\nyour gender: ')
        if gender in GENDERS:
            break
    while True:
        prof = input('professions: ' + re.compile("(['\]\[])").sub(r'', str(PROFFS)) + '\nyour profession: ')
        if prof in PROFFS:
            break
    while True:
        genre1 = input('genres: ' + re.compile("(['\]\[])").sub(r'', str(GENRES)) + '\nyour first genre: ')
        if genre1 in GENRES:
            break
    while True:
        genre2 = input('genres: ' + re.compile("(['\]\[])").sub(r'', str(GENRES)) + '\nyour second genre: ')
        if genre2 in GENRES:
            break

    user_data = [[age], [lang], [gender], [prof], [genre1, genre2]]
    print(f'your data: {user_data}')
else:

    user_data = [[22], ['it'], ['M'], ['programmer'], ['crime', 'drama']]
    print(user_data)

print_recommendation(user_data)
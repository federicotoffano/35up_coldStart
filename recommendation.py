import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from pickle import load
import numpy as np
from tensorflow import keras

df_items_formatted = pd.read_csv('db/items_formatted.csv')
df_users_formatted = pd.read_csv('db/users_formatted.csv')
df_users = pd.read_csv('db/users.csv')
df_items = pd.read_csv('db/items.csv')
df_gens = pd.read_csv('db/genres.csv')

genre_dict = dict(zip(df_gens['id'], df_gens['genre']))

print(df_users_formatted.columns)
print(print(df_users.columns))


bin_list = []
with open(r"model/user_age_binarizer.pkl", "rb") as input_file:
    user_age_binarizer = load(input_file)
    bin_list.append(user_age_binarizer)
with open(r"model/user_nationality_binarizer.pkl", "rb") as input_file:
    user_nationality_binarizer = load(input_file)
    bin_list.append(user_nationality_binarizer)
with open(r"model/user_gender_binarizer.pkl", "rb") as input_file:
    user_gender_binarizer = load(input_file)
    bin_list.append(user_gender_binarizer)
with open(r"model/user_occupation_binarizer.pkl", "rb") as input_file:
    user_occupation_binarizer = load(input_file)
    bin_list.append(user_occupation_binarizer)
with open(r"model/genres_binarizer.pkl", "rb") as input_file:
    genres_binarizer = load(input_file)
    bin_list.append(genres_binarizer)


# sample_input = [['Under18'], ['it'], ['M'], ['programmer'], ['animation', 'family']]
# sample_input = [['25-34'], ['de'], ['M'], ['programmer'], ['music', 'western']]
# sample_input = [['Under18'], ['it'], ['M'], ['programmer'], ['animation', 'adventure']]
# sample_input = [['35+'], ['it'], ['M'], ['comedy'], ['history', 'family']]
sample_input = [['18-24'], ['de'], ['M'], ['comedy'], ['crime', 'mystery']]



user_vector = np.array([])
print(user_vector)
for index, val in enumerate(sample_input):
    vec = bin_list[index].transform([val])[0]
    print(vec)
    user_vector = np.concatenate((user_vector, vec))
# print(user_vector)
model = keras.models.load_model('model/seq.keras')
user_item_list = np.array([np.array(np.concatenate((user_vector, np.array(row[1:]).tolist())))
                                            for index, row in df_items_formatted.iterrows()])
test_scores = model.predict(user_item_list).flatten()

print(test_scores)
print(max(test_scores))

n_top_items = 5
ind = np.argpartition(test_scores, -n_top_items)[-n_top_items:]
top_items = df_items.iloc[ind]
top_test_scores = test_scores[ind]
recc = 0
genre_dict = dict(zip(df_gens['id'], df_gens['genre']))
for index, row in top_items.iterrows():
    gens_indexes = str(row['genres']).split('-')
    gen_names = [genre_dict[int(gen)] for gen in gens_indexes]
    original_title = str(row['original_title'])
    original_language = str(row['original_language'])
    year = str(row['year'])
    popularity  = str(row['popularity'])
    print(f'\nRecommendation {recc+1}'
          f'\nTitle: {original_title}'
          f'\nGenres: {gen_names}'
          f'\nOriginal Language: {original_language}'
          f'\nYear: {year}'
          f'\nPopularity: {popularity}'
          f'\nConfidence of recommendation: {top_test_scores[recc]}')
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


import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from pickle import dump

def binarizer(colname, df, prefix_path):
    df_temp = df.copy()
    ml_binarizer = MultiLabelBinarizer()
    to_trans = ml_binarizer.fit_transform([[x] for x in list(df_temp[colname])])
    bin_ds = pd.DataFrame(to_trans, columns=ml_binarizer.classes_)
    df_temp = pd.concat([df_temp, bin_ds], axis=1)
    df_temp.drop(colname, axis=1, inplace=True)
    dump(ml_binarizer, open(f'{prefix_path}_binarizer.pkl', 'wb'))
    return df_temp

def scaler(colname, df, prefix_path):
    df_temp = df.copy()
    min_max_scaler = MinMaxScaler()
    res = min_max_scaler.fit_transform(pd.DataFrame(df_temp[colname]))
    df_temp[colname] = pd.DataFrame(res, columns=[colname])
    dump(min_max_scaler, open(f'{prefix_path}_scaler.pkl', 'wb'))
    return df_temp


def format():
    df_gens = pd.read_csv('db/genres.csv')
    genre_dict = dict(zip(df_gens['id'], df_gens['genre']))
    genres_ml_binarizer = MultiLabelBinarizer()
    genres_ml_binarizer.fit_transform([list(df_gens['genre'])])
    dump(genres_ml_binarizer, open('model/genres_binarizer.pkl', 'wb'))

    #--------------MOVIES----------------
    df_items = pd.read_csv('db/items.csv')
    df_items.drop('original_title', axis=1, inplace=True)

    # One-hot encoding of genres

    #id and binarizied categories for each movie
    genres_items_bindata = []
    for index, row in df_items.iterrows():
        gens = str(row['genres']).split('-')
        to_trans = [genre_dict[int(gen)] for gen in gens]
        res = list(genres_ml_binarizer.transform([to_trans])[0])
        genres_items_bindata.append(res)
    bin_ds = pd.DataFrame(genres_items_bindata, columns=genres_ml_binarizer.classes_)
    df_items = pd.concat([df_items, bin_ds], axis=1)

    df_items.drop('genres', axis=1, inplace=True)

    #rescale numeric columns
    for col in ['popularity', 'year', 'revenue', 'vote_average', 'vote_count']:
        df_items = scaler(col, df_items, f'model/item_{col}')

    #one-hot encoding languages
    df_items = binarizer('original_language', df_items, f'model/item_original_language')

    df_items.to_csv('db/items_formatted.csv', index=False)


    #--------------USERS----------------
    df_users = pd.read_csv('db/users.csv')
    #one-hot encoding
    for col in ['age', 'nationality', 'gender', 'occupation']:
        df_users = binarizer(col, df_users, f'model/user_{col}')

    # One-hot encoding of genres
    genres_users_bindata = []
    for index, row in df_users.iterrows():
        gens = [int(row['genre1']), int(row['genre2'])]
        to_trans = [genre_dict[int(gen)] for gen in gens]
        res = list(genres_ml_binarizer.transform([to_trans])[0])
        # res.extend([int(row['id'])])
        genres_users_bindata.append(res)
    # df_genres_bindata = pd.DataFrame(genres_users_bindata, columns=['id']+list(genres_labelBinarizer.classes_))
    # df_users = pd.merge(df_users, df_genres_bindata, on='id')

    bin_ds = pd.DataFrame(genres_users_bindata, columns=genres_ml_binarizer.classes_)
    df_users = pd.concat([df_users, bin_ds], axis=1)

    df_users.drop('genre1', axis=1, inplace=True)
    df_users.drop('genre2', axis=1, inplace=True)

    df_users.to_csv('db/users_formatted.csv', index=False)

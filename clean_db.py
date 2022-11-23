import pandas as pd
import json
import difflib
import numpy as np
from typing import Union, List

#look for simialr items using difflib.SequenceMatcher
#it takes time...
SIMIL_ITEMS_SEARCH = False

Num = Union[int, float]
#log variable
log = ''


def get_val(df: pd.DataFrame, row_index: int, col_name: str):
    assert type(col_name) is str
    assert type(row_index) is int
    assert type(df) is pd.DataFrame
    """
    Get dataframe value at [row_index, col_name]
    :param df: pandas.DataFrame: DataFrame
    :param row_index: int: index of row
    :param col_name: string: column name
    :return: None if empty cell or value at [row_index, col_name]
    """
    global log

    cell = df.loc[row_index, col_name]
    if pd.isna(cell) or cell == '' or cell == '[]':
        msg = f'Empty {col_name} at row {row_index}'
        print(msg)
        log += f'\n{msg}'
        return None
    else:
        return row[col_name]

def check_range(val: Num, rng: List[Num], row_index: int, col_name: str) -> bool:
    assert len(rng) == 2
    assert type(val) is float or type(val) is int
    assert min([type(x) is float or type(x) is int for x in rng])==1
    assert type(col_name) is str
    assert type(row_index) is int
    """
    Check if val >= rng[0] and val <= rng[1]. If not, print error message for row_index and col_name
    :param val: float or int: value to check
    :param rng: list of float or int: upper and lower bound.
    :param row_index: int: index of row
    :param col_name: string: column name
    :return: boolean: True if val is in range, False otherwise.
    """
    global log
    if val < rng[0] or val > rng[1]:
        msg = f'{col_name}({year}) not in range at row {row_index}'
        print(msg)
        log += f'\n{msg}'
        return False
    return True


df = pd.read_csv('db/tmdb_5000_movies.csv')
df = df.reset_index()  # make sure indexes pair with number of rows
print(list(df.columns))

#Data extraction: genres,id,original_language,original_title,
#                popularity,year,revenue,vote_average,vote_count

items_df = pd.DataFrame({'id': pd.Series(dtype='int'),
                   'genres': pd.Series(dtype='str'),
                   'original_language': pd.Series(dtype='str'),
                   'original_title': pd.Series(dtype='str'),
                   'popularity': pd.Series(dtype='float'),
                    'year': pd.Series(dtype='int'),
                    'revenue': pd.Series(dtype='float'),
                    'vote_average': pd.Series(dtype='float'),
                    'vote_count': pd.Series(dtype='int')})

genres_df = pd.DataFrame({'id': pd.Series(dtype='int'),
                   'genre': pd.Series(dtype='str')})

dropped_rows = 0

for row_index, row in df.iterrows():
    id = int(get_val(df, row_index, 'id'))
    if id is None:
        dropped_rows += 1
        continue
    if id in set(items_df['id']):
        dropped_rows += 1
        msg = f'Duplicated item ID \'{id}\' at row {row_index}'
        log += f'\n{msg}'
        print(msg)
        continue

    gens = get_val(df, row_index, 'genres')
    if gens is None:
        dropped_rows += 1
        continue
    gens_json = json.loads(gens)
    gen_id_list = np.array([], dtype='int32')
    gen_name_list = np.array([])
    for g in gens_json:
        gen_id_list = np.append(gen_id_list, int(g['id']))
        gen_name_list = np.append(gen_name_list, g['name'].lower())
    #sorted categories
    inds = gen_id_list.argsort()
    gen_name_list = gen_name_list[inds]
    gen_id_list.sort()
    gens_id_str = '-'.join(map(str, gen_id_list))

    original_language = get_val(df, row_index, 'original_language')
    if original_language is None:
        continue
    original_language = original_language.lower()

    original_title = get_val(df, row_index, 'original_title')
    if original_title is None:
        dropped_rows += 1
        continue
    original_title = original_title.lower()

    if original_title in set(items_df['original_title']):
        dropped_rows += 1
        msg = f'Duplicated item title \'{original_title}\' at row {row_index}'
        log += f'\n{msg}'
        print(msg)
        continue

    popularity = get_val(df, row_index, 'popularity')
    if popularity is None or not check_range(popularity, [0, float('inf')], row_index, 'popularity'):
        dropped_rows += 1
        continue

    release_date = get_val(df, row_index, 'release_date')
    if release_date is None:
        dropped_rows += 1
        continue
    year = int(release_date.split('-')[0])
    if not check_range(year, [1895, 2022], row_index, 'release_date'):
        dropped_rows += 1
        continue

    revenue = get_val(df, row_index, 'revenue')
    if revenue is None:
        dropped_rows += 1
        continue
    if revenue < 0 or not check_range(revenue, [0, 10**11], row_index, 'revenue'):
        dropped_rows += 1
        continue

    vote_average = get_val(df, row_index, 'vote_average')
    if vote_average is None or not check_range(vote_average, [0, float('inf')], row_index, 'vote_average'):
        dropped_rows += 1
        continue

    vote_count = get_val(df, row_index, 'vote_count')
    if vote_count is None or not check_range(vote_count, [0, float('inf')], row_index, 'vote_count'):
        dropped_rows += 1
        continue

    item_row = {'id': id,
                   'genres': gens_id_str,
                   'original_language': original_language,
                   'original_title': original_title,
                   'popularity': popularity,
                    'year': year,
                    'revenue': revenue,
                    'vote_average': vote_average,
                    'vote_count': vote_count}

    #update items df
    items_df = pd.concat([items_df, pd.DataFrame([item_row])], axis=0, ignore_index=True)

    #updated genres df
    for i in range(len(gen_id_list)):
        if gen_id_list[i] not in set(genres_df['id']):
            genre_row = {'id': gen_id_list[i], 'genre': gen_name_list[i]}
            genres_df = pd.concat([genres_df, pd.DataFrame([genre_row])], axis=0, ignore_index=True)

#check if there are duplicates
# unique_genres = set(genres_df['genre'])
# print(f'Duplicated categories: {len(genres_df) != len(unique_genres)}')
genres_found = ' '.join('{}({})'.format(a, b) for a, b in zip(genres_df['genre'], genres_df['id']))
print(f'Genres found <name(ID)>: {genres_found}')
print(f'Dropped rows: {dropped_rows}')
print(f'New Dataset length: {len(items_df)}')

items_df.to_csv('db/items.csv', index=False)
genres_df.to_csv('db/genres.csv', index=False)


if SIMIL_ITEMS_SEARCH:
    # manual check to see if there are duplicates
    log += "\nLooking for similar items..."
    print('\nLooking for similar items...')
    items_df.drop('id', inplace=True, axis=1)
    items_df.drop('popularity', inplace=True, axis=1)
    items_df.drop('vote_average', inplace=True, axis=1)
    items_df.drop('vote_count', inplace=True, axis=1)
    #create string representing row
    r = items_df.to_string(header=False, index=False, index_names=False).split('\n')
    vals = ['-'.join(el.split()) for el in r]

    for index1, t1 in enumerate(vals):
        for index2 in range(index1 + 1, len(vals)):
            t2 = vals[index2]
            # T - total number of elements in both strings (len(first_string) + len(second_string))
            # M - number of matches
            # Distance = 2.0 * M / T -> between 0.0 and 1.0
            # (1.0 if the sequences are identical, and 0.0 if they don't have anything in common)
            seq = difflib.SequenceMatcher(a=t1, b=t2)
            if seq.ratio() > 0.9:
                msg = f'\nindexes {index1}, {index2}:\n{t1}\n{t2}'
                log += f'\n{msg}'
                print(msg)

text_file = open("db/log.txt", "w")
text_file.write(log)
text_file.close()


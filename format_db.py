import pandas as pd

df_gens = pd.read_csv('db/genres.csv')
genre_dict = dict(zip(df_gens['id'], df_gens['genre']))

#--------------MOVIES----------------
df_movies = pd.read_csv('db/items.csv')
# One-hot encoding of genres
for val in genre_dict.values():
    df_movies.insert(len(df_movies.columns), val, 0, allow_duplicates=True)
for index, row in df_movies.iterrows():
    gens = str(row['genres']).split('-')
    for gen in gens:
        df_movies.at[index,genre_dict[int(gen)]]=1

df_movies.drop('genres', axis=1, inplace=True)

#rescale numeric columns
df_movies[['popularity','year','revenue','vote_average','vote_count']] -= \
    df_movies[['popularity','year','revenue','vote_average','vote_count']].min()
df_movies[['popularity','year','revenue','vote_average','vote_count']] /= \
    df_movies[['popularity','year','revenue','vote_average','vote_count']].max()
#one-hot encoding languages
df_movies = pd.get_dummies(df_movies,prefix=['original_language'],
                           columns = ['original_language'], drop_first=True)
df_movies.drop('original_title', axis=1, inplace=True)

df_movies.to_csv('db/items_formatted.csv', index=False)

#--------------USERS----------------
df_users = pd.read_csv('db/users.csv')
#one-hot encoding
df_users = pd.get_dummies(df_users,prefix=['age'],
                          columns = ['age'], drop_first=True)
df_users = pd.get_dummies(df_users,prefix=['nationality'],
                          columns = ['nationality'], drop_first=True)
df_users = pd.get_dummies(df_users,prefix=['gender'],
                          columns = ['gender'], drop_first=True)
df_users = pd.get_dummies(df_users,prefix=['occupation'],
                          columns = ['occupation'], drop_first=True)

# One-hot encoding of genres
for val in genre_dict.values():
    df_users.insert(len(df_users.columns), val, 0, allow_duplicates=True)
for index, row in df_users.iterrows():
    gens = [int(row['genre1']), int(row['genre2'])]
    for gen in gens:
        df_users.at[index,genre_dict[gen]]=1
df_users.drop('genre1', axis=1, inplace=True)
df_users.drop('genre2', axis=1, inplace=True)

df_users.to_csv('db/users_formatted.csv', index=False)

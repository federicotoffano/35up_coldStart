
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
rnd.seed(0)

#fraction of of random data
NOISE_USERS = 0.3
NOISE_CLICKS = 0.1

#number of users. We generate 1 positive click and 4 negative click per user
N_USERS = 2500

#user parameters
AGES =  ["Under18", "18-24", "25-34", "35+"]
NATIONALITIES = ['it', 'pt', 'de']
GENDERS = ['M', 'F']
OCCUPATIONS = ['tradesman', "programmer", "self-employed"]
USER_FEATURES = ['age', 'nationality', 'gender', 'occupation', 'genre1',  'genre2']

# Genres found <name(ID)>: adventure(12) fantasy(14) action(28)
# science fiction(878) crime(80) drama(18) thriller(53)
# animation(16) family(10751) western(37) comedy(35)
# romance(10749) horror(27) mystery(9648) history(36)
# war(10752) music(10402) documentary(99) foreign(10769) tv movie(10770)


def generate_users(df_gens: pd.DataFrame) -> pd.DataFrame:
    """
    with a noise of probability p:
    if age == 'Under18':
        genre 1: 1-p/2 animation, 1-p/2 adventure, p others
        genre 2: 1-p/2 comedy, 1-p/2 family, p others
    if age == "18-24":
        italians: genre1 and genre2 p crime and drama, p others
        brazilians: genre1 and genre2 p science fiction and action, p others
        germans: genre1 and genre2 p horror and mystery, p others
    if age == "25-34":
        italians: 1-p documentary and western, p others
        brazilians: 1-p history fiction and western p others
        germans: 1-p music and western p others
    if age == '35+':
        genre 1: 1-p/2 documentary, 1-p/2 history, p others
        genre 2: 1-p/2 comedy, 1-p/2 family, p others

    :param df_gens: pandas.Dataframe: genres dataframe
    :return: pandas.Dataframe: users dataframe
    """

    genre_dict = dict(zip(df_gens['genre'], df_gens['id']))
    df_users = pd.DataFrame({'id': pd.Series(dtype='int'),
                             'age': pd.Series(dtype='int'),
                             'nationality': pd.Series(dtype='str'),
                             'gender': pd.Series(dtype='str'),
                             'occupation': pd.Series(dtype='str'),
                             'genre1': pd.Series(dtype='int'),
                             'genre2': pd.Series(dtype='int')})

    for i in range(N_USERS):
        age = str(rnd.sample(AGES, 1)[0])
        nationality = rnd.sample(NATIONALITIES, 1)[0]
        gender = rnd.sample(GENDERS, 1)[0]
        occupation = rnd.sample(OCCUPATIONS, 1)[0]

        #probability genre 1
        gen_1_p = (1.0-NOISE_USERS)/2
        #probability genre 2
        gen_2_p = (1.0-NOISE_USERS)/2
        if age == 'Under18':
            # genre 1 1-p/2 animation, 1-p/2 adventure, 20% others
            weights1 = [NOISE_USERS / (len(genre_dict)-2)] * len(genre_dict)
            weights1[list(genre_dict.keys()).index("animation")] = gen_1_p
            weights1[list(genre_dict.keys()).index("adventure")] = gen_2_p
            genre1_name = rnd.choices(list(genre_dict.keys()), weights = weights1)[0]

            # genre 2 1-p/2 comedy, 1-p/2 family, 20% others
            # new_genre_dict = dict((i, genre_dict[i]) for i in genre_dict if i != genre1_name)
            weights2 = [NOISE_USERS / (len(genre_dict)-2)] * len(genre_dict)
            weights2[list(genre_dict.keys()).index("comedy")] = gen_1_p
            weights2[list(genre_dict.keys()).index("family")] = gen_2_p
            genre2_name = rnd.choices(list(genre_dict.keys()), weights = weights2)[0]

        elif age == "18-24":
            if rnd.random() > NOISE_USERS:
                #italians: 80% crime and drama, 20% others
                if nationality == 'it':
                    genre1_name = 'crime'
                    genre2_name = 'drama'
                #brazilians: 80% science fiction and action 20% others
                elif nationality == 'pt':
                    genre1_name = 'science fiction'
                    genre2_name = 'action'
                #germans: 80% horror and mystery 20% others
                else:
                    genre1_name = 'horror'
                    genre2_name = 'mystery'
            else:
                genre1_name, genre2_name = rnd.sample(list(genre_dict.keys()), 2)

        elif age == "25-34":
            if rnd.random() > NOISE_USERS:
                #italians: 1-p documentary and western, 20% others
                if nationality == 'it':
                    genre1_name = 'documentary'
                    genre2_name = 'western'
                #brazilians: 1-p history fiction and western 20% others
                elif nationality == 'pt':
                    genre1_name = 'history'
                    genre2_name = 'western'
                #germans: 1-p music and western p others
                else:
                    genre1_name = 'music'
                    genre2_name = 'western'
            else:
                genre1_name, genre2_name = rnd.sample(list(genre_dict.keys()), 2)

        else:
            # genre 1 1-p/2 documentary, 1-p/2 history, p others
            weights1 = [NOISE_USERS / (len(genre_dict)-2)] * len(genre_dict)
            weights1[list(genre_dict.keys()).index("documentary")] = gen_1_p
            weights1[list(genre_dict.keys()).index("history")] = gen_2_p
            genre1_name = rnd.choices(list(genre_dict.keys()), weights = weights1)[0]

            # genre 2 1-p/2 comedy, 1-p/2 family, p others
            weights2 = [NOISE_USERS / (len(genre_dict)-2)] * len(genre_dict)
            weights2[list(genre_dict.keys()).index("comedy")] = gen_1_p
            weights2[list(genre_dict.keys()).index("family")] = gen_2_p
            genre2_name = rnd.choices(list(genre_dict.keys()), weights = weights2)[0]

        sampled_gens_id = [genre_dict[genre1_name], genre_dict[genre2_name]]
        #othersize genres order
        rnd.shuffle(sampled_gens_id)
        user = {'id': i,
                    'age': age,
                    'nationality': nationality,
                    'gender': gender,
                    'occupation': occupation,
                    'genre1': sampled_gens_id[0],
                    'genre2': sampled_gens_id[1]}

        #insert row
        df_users = pd.concat([df_users, pd.DataFrame([user])], axis=0, ignore_index=True)

    return df_users

def generate_interactions(df_users: pd.DataFrame, df_items: pd.DataFrame) -> pd.DataFrame:
    """
    5 interactions per user, only one was clicked.
    Clicking criteria: movie is of genre1 (condition 1),
    and year > 2010 (condition 2) if Under18
    and popularity > mean popularity (condition 2) if 18-24
    and original_language = user language (condition 2) if 25-34
    and movie is of genre2 (condition 2) if 35+

    :param df_users: pandas.Dataframe: users dataframe
    :param df_items: pandas.Dataframe: items dataframe
    :return: pandas.Dataframe: interaction user-time dataframe
    """

    #item header:
    #id,genres,original_language,original_title,popularity,year,revenue,vote_average,vote_count

    df_interactions = pd.DataFrame({'userID': pd.Series(dtype='int'),
                             'itemID': pd.Series(dtype='int'),
                             'interaction': pd.Series(dtype='bool')})

    for index, user in df_users.iterrows():
        # store 1 positive interaction and 4 negative interactions.
        neg_answers = 4

        if rnd.random() > NOISE_CLICKS:
            if user['age'] == 'Under18':
                #movie is of genre1 (condition 1)
                df_items_tmp = df_items[df_items['genres'].str.contains(str(user['genre1']))]
                df_items_tmp_neg = df_items[~df_items['genres'].str.contains(str(user['genre1']))]
                #and year > 2010
                df_items_tmp_2 = df_items_tmp[df_items_tmp['year'] > 2010]
                df_items_tmp_2_neg = df_items_tmp_neg[df_items_tmp_neg['year'] <= 2010]

            elif user['age'] == "18-24":
                #movie is of genre1 (condition 1)
                df_items_tmp = df_items[df_items['genres'].str.contains(str(user['genre1']))]
                df_items_tmp_neg = df_items[~df_items['genres'].str.contains(str(user['genre1']))]
                #and popularity > mean popularity
                df_items_tmp_2 = df_items_tmp[df_items_tmp['popularity'] > df_items_tmp["popularity"].mean()]
                df_items_tmp_2_neg = df_items_tmp_neg[df_items_tmp_neg['popularity'] <= df_items_tmp["popularity"].mean()]

            elif user['age'] == "25-34":
                #movie is of genre1 (condition 1)
                # df_items_tmp = df_items[df_items['genres'].str.contains(str(user['genre1']))]
                # df_items_tmp_neg = df_items[~df_items['genres'].str.contains(str(user['genre1']))]
                # #and original_language = user language
                # df_items_tmp_2 = df_items_tmp[df_items_tmp['original_language'].str.contains(str(user['nationality']))]
                # df_items_tmp_2_neg = df_items_tmp_neg[~df_items_tmp_neg['original_language'].str.contains(str(user['nationality']))]
                df_items_tmp = df_items[df_items['original_language'].str.contains(str(user['nationality']))]
                df_items_tmp_neg = df_items[~df_items['original_language'].str.contains(str(user['nationality']))]
                df_items_tmp_2 = df_items_tmp
                df_items_tmp_2_neg = df_items_tmp_neg
            else:#elif user['age'] == "35+":
                #movie is of genre1 (condition 1)
                df_items_tmp = df_items[df_items['genres'].str.contains(str(user['genre1']))]
                df_items_tmp_neg = df_items[~df_items['genres'].str.contains(str(user['genre1']))]
                #and movie is of genre2
                df_items_tmp_2 = df_items_tmp[df_items_tmp['genres'].str.contains(str(user['genre2']))]
                df_items_tmp_2_neg = df_items_tmp_neg[~df_items_tmp_neg['genres'].str.contains(str(user['genre2']))]

            # store positive interaction

            # len(df_items_tmp_2) > 0 if condition1 and condition 2 are satisfied.
            if len(df_items_tmp_2) > 0:
                id_pos = int(df_items_tmp_2.sample()['id'])
            # Otherwise focus only on condition1
            else:
                id_pos = int(df_items_tmp.sample()['id'])

            if len(df_items_tmp_2_neg) > neg_answers:
                df_items_neg = df_items_tmp_2_neg
            else:
                df_items_neg = df_items_tmp_neg
        else:
            id_pos = int(df_items.sample()['id'])
            df_items_neg = df_items[(df_items['id'] != id_pos)]
        interaction = {'userID': int(user['id']), 'itemID': id_pos, 'interaction': True}
        df_interactions = pd.concat([df_interactions, pd.DataFrame([interaction])], axis=0, ignore_index=True)

        # A random movie different than the movie of the positive interaction
        ids_neg = [int(el) for el in df_items_neg.sample(neg_answers)['id']]
        for id_neg in ids_neg:
            interaction = {'userID': int(user['id']), 'itemID': id_neg, 'interaction': False}
            df_interactions = pd.concat([df_interactions, pd.DataFrame([interaction])], axis=0, ignore_index=True)

    return df_interactions

df_gens = pd.read_csv('db/genres.csv')
df_users = generate_users(df_gens)
df_users.to_csv('db/users.csv', index=False)
df_items = pd.read_csv('db/items.csv')
df_interactions = generate_interactions(df_users, df_items)
df_interactions.to_csv('db/interactions.csv', index=False)


df_interactions = pd.merge(df_interactions, df_users.rename(columns={'id': 'userID'}), on='userID')
df_interactions = pd.merge(df_interactions, df_items.rename(columns={'id': 'itemID'}), on='itemID')
df_interactions.drop('userID', axis=1, inplace=True)
df_interactions.drop('itemID', axis=1, inplace=True)

for age in set(df_interactions['age']):
    df_interactions_tmp = df_interactions[df_interactions['age'] == age]
    for nat in set(df_interactions_tmp['nationality']):
        df_interactions_tmp2 = df_interactions_tmp[df_interactions_tmp['interaction'] == True]
        df_interactions_tmp2 = df_interactions_tmp2[df_interactions_tmp['nationality'] == nat]
        # d = df_interactions_tmp['original_language']
        d = pd.DataFrame({'s1': df_interactions_tmp2['original_language']})
        d.apply(pd.value_counts).plot(kind='bar', subplots=True,
                                      title=f'Positive click '
                                            f'of user nationality({nat}) and age({age}) (y)\n'
                                            f'w.r.t. movie language (x)')
        plt.show(block=True)

# plot histogram of genre1 and genre2 by age
for age in set(df_users['age']):
    df_users_tmp = df_users[df_users['age'] == age]
    gen1 = df_users_tmp['genre1']
    gen2 = df_users_tmp['genre2']
    d = pd.DataFrame({'genre1': gen1, 'genre2': gen2})
    d.apply(pd.value_counts).plot(kind='bar', subplots=True, title=f'Number of user '
                                            f'with  age({age}) (y)\n'
                                            f'w.r.t. genres (x)')
    plt.show(block=True)


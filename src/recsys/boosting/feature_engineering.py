import pandas as pd


def get_engineering_features(data, user_features_data, item_features_data):
    r"""Make feature engineering of user-based and item-based features over provided data.

    Calculates additional features.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with 'user_id', 'item_id' columns.
    user_features_data : pd.DataFrame
        User-based features.
    item_features_data : pd.DataFrame
        Item-based features.

    Returns
    -------
    user_feature_engineering_df : pd.DataFrame
        New user-based features.
    item_feature_engineering_df : pd.DataFrame
        New item-based features.
    """
    features_df = data[['user_id', 'item_id', 'timestamp', 'rating']]
    features_df = pd.merge(features_df, user_features_data, on='user_id', how='left')
    features_df = pd.merge(features_df, item_features_data, on='item_id', how='left')

    # Кол-во просмотров у юзера + средний рейтинг
    tmp = features_df.groupby(
        'user_id',
        as_index=False
    ).agg(
        {'item_id':'count', 'rating': 'mean'}
    ).rename(columns={'item_id': 'user_count_int', 'rating': 'user_mean_rating'})
    features_df = pd.merge(features_df, tmp, on='user_id', how='inner')

    # Среднее время между взаимодействиями
    features_df['delta_with_prev'] = features_df['timestamp'] - features_df.groupby('user_id', as_index=False)['timestamp'].shift(1).values
    tmp = features_df.groupby(
        'user_id',
        dropna=True, 
        as_index=False
    ).agg(
        {'delta_with_prev': 'mean'}
    )
    features_df = pd.merge(features_df.drop(columns='delta_with_prev'), tmp, on='user_id', how='inner')

    # Кол-во взаимодействий с каждым из жанров
    num_genres = sum(['genre' in column for column in item_features_data.columns])
    for i in range(num_genres):
        tmp = features_df.groupby(
            'user_id',
            as_index=False
        ).agg(
            {f'genre_{i}': 'sum'}
        ).rename(columns={f'genre_{i}': f'user_int_{i}'})
        features_df = pd.merge(features_df, tmp, on='user_id', how='inner')
    
    # Cредний рейтинг по каждому из жанров
    # Если не смотрел - NaN
    for i in range(num_genres):
        tmp = features_df[features_df[f'genre_{i}'] != 0].groupby(
            'user_id', 
            as_index=False
        ).agg(
            {'rating': 'mean'}
        ).rename(columns={f'rating': f'user_mean_rating_{i}'})
        features_df = pd.merge(features_df, tmp, on='user_id', how='left')
    
    user_features = ['gender', 'age', 'user_count_int', 'user_mean_rating', 'delta_with_prev']\
                    + [f'user_int_{i}' for i in range(num_genres)]\
                    + [f'user_mean_rating_{i}' for i in range(num_genres)]

    # Сколько раз оценивали этот фильм + средний рейтинг
    tmp = features_df.groupby(
        'item_id',
        as_index=False
    ).agg(
        {'user_id': 'count', 'rating': 'mean'}
    ).rename(columns = {'user_id': 'item_popularity', 'rating': 'item_mean_rating'})
    features_df = pd.merge(features_df, tmp, on='item_id', how='inner')

    # Сколько жанров у фильма
    genre_cols = [f'genre_{i}' for i in range(num_genres)]
    features_df['item_genre_cnt'] = features_df[genre_cols].sum(axis=1).values

    # Среднее время между покупками этого айтема
    features_df['item_delta_with_prev'] = features_df['timestamp'] - features_df.groupby('item_id', as_index=False)['timestamp'].shift(1).values
    tmp = features_df.groupby(
        'item_id', 
        dropna=True, 
        as_index=False
    ).agg({'item_delta_with_prev': 'mean'})
    features_df = pd.merge(features_df.drop(columns='item_delta_with_prev'), tmp, on='item_id', how='inner')

    item_features = [f'genre_{i}' for i in range(num_genres)] + ['item_popularity', 'item_mean_rating', 'item_genre_cnt', 'item_delta_with_prev']

    user_feature_engineering_df = features_df[['user_id'] + user_features].drop_duplicates()
    item_feature_engineering_df = features_df[['item_id'] + item_features].drop_duplicates()

    return user_feature_engineering_df, item_feature_engineering_df

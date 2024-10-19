import pandas as pd


def sample_random_negatives(positives_df, n_items, n_negatives=250):
    r"""Make random negative sampling.

    Random negative sampling.

    Parameters
    ----------
    positives_df : pd.DataFrame
        Dataframe of positive examples with 'user_id', 'item_id' columns.
    n_items : int
        Number of total items.
    n_negatives : int
        Number of negatives for each user.

    Returns
    -------
    positives_df : pd.DataFrame
        Dataframe of negative examples with 'user_id', 'item_id' columns.
    """
    negatives_df = positives_df.groupby('user_id')['item_id'].apply(
        lambda _: list(range(n_items))
    ).reset_index().explode('item_id')
    negatives_df = pd.merge(
        negatives_df,
        positives_df[['user_id', 'item_id']].assign(__tmp__=True),
        on=['user_id', 'item_id'],
        how='left'
    )
    negatives_df = negatives_df[negatives_df['__tmp__'].isna()].drop('__tmp__', axis=1).sample(frac=1, random_state=777)
    # n negatives per each user
    negatives_df = negatives_df.groupby('user_id').head(n_negatives).reset_index(drop=True)
    negatives_df['target'] = 0
    return negatives_df

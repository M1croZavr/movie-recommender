from implicit.als import AlternatingLeastSquares
from implicit.cpu.bpr import BayesianPersonalizedRanking
from implicit.cpu.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import CosineRecommender
from scipy.sparse import csr_matrix


def train_candidates_models(data, n_users, n_items):
    r"""Make instances and train candidates scoring models.

    Make models and train by provided data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with 'user_id', 'item_id' and 'weight' columns.
    n_users : int
        Number of total users.
    n_items : int
        Number of total items.

    Returns
    -------
    models : list
        List of trained models which support trained vectors.
    scores : list
        List of user/item score for other models.
    """
    # Convert to Compressed Sparse Row matrix
    candidates_als_csr = csr_matrix(
        (
            data['weight'],
            (data['user_id'], data['item_id'])
        ),
        shape=(n_users, n_items)
    )
    candidates_csr = csr_matrix(
        (
            data.loc[data['weight'] > 0, 'weight'],
            (
                data.loc[data['weight'] > 0, 'user_id'],
                data.loc[data['weight'] > 0, 'item_id']
            )
        ),
        shape=(n_users, n_items)
    )

    # Create instances and train models
    model_als = AlternatingLeastSquares(
        factors=128, regularization=0.05, alpha=1.5, iterations=50, random_state=777, calculate_training_loss=True
    )
    model_als.fit(candidates_als_csr)

    model_bpr = BayesianPersonalizedRanking(factors=64, iterations=200, random_state=777)
    model_bpr.fit(candidates_csr)

    model_lmf = LogisticMatrixFactorization(factors=32, iterations=60, random_state=777)
    model_lmf.fit(candidates_csr)

    model_cosine = CosineRecommender(50)
    model_cosine.fit(candidates_csr.astype(float))
    userids = data['user_id'].unique()
    nn_ids, nn_scores = model_cosine.recommend(
        userids, 
        candidates_csr.astype(float)[userids],
        N=n_items,
        filter_already_liked_items=False
    )
    
    return [model_als, model_bpr, model_lmf], [nn_scores]

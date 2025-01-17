from .ml_base import MLBaseDataset


class ML100KDataset(MLBaseDataset):
    @classmethod
    def code(cls):
        return 'ml-100k'

    @classmethod
    def url(cls):  # as of Sep 2023
        return 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

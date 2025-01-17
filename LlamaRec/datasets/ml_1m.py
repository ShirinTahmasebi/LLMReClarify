from .ml_base import MLBaseDataset
import numpy as np
import pandas as pd
import re

class ML1MDataset(MLBaseDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

    
    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, delimiter='::')
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    
    
    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('movies.dat')
        df = pd.read_csv(file_path, delimiter='::', header=None, encoding="ISO-8859-1")
        meta_dict = {}
        for row in df.itertuples():
            title = row[2][:-7]  # remove year (optional)
            year = row[2][-7:]

            title = re.sub('\(.*?\)', '', title).strip()
            # the rest articles and parentheses are not considered here
            if any(', '+x in title.lower()[-5:] for x in ['a', 'an', 'the']):
                title_pre = title.split(', ')[:-1]
                title_post = title.split(', ')[-1]
                title_pre = ', '.join(title_pre)
                title = title_post + ' ' + title_pre

            meta_dict[row[1]] = title + year
        return meta_dict


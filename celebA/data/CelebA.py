import os
import pandas as pd
from tqdm import tqdm

from celebA.config import BUCKET_NAME, image_path, partition_path, att_path
from celebA.utils.gcs_utils import _download


class CelebA():
    '''Wraps the celebA dataset, allowing an easy way to:
         - Select the features of interest,
         - Split the dataset into 'training', 'test' or 'validation' partition.
    '''

    def __init__(self, main_folder='celeba-dataset/', selected_features=None, drop_features=[]):
        self.main_folder = main_folder
        self.images_folder = os.path.join(os.getenv("HOME"), "Tensorleap_data", BUCKET_NAME, image_path)
        # self.images_folder = image_path
        self.attributes_path = _download(att_path)
        self.partition_path = _download(partition_path)
        self.selected_features = selected_features
        self.features_name = []
        self.__prepare(drop_features)


    def __prepare(self, drop_features):
        '''do some preprocessing before using the data: e.g. feature selection'''
        # attributes:
        if self.selected_features is None:
            self.attributes = pd.read_csv(self.attributes_path)
            self.num_features = 40
        else:
            self.num_features = len(self.selected_features)
            self.selected_features = self.selected_features.copy()
            self.selected_features.append('image_id')
            self.attributes = pd.read_csv(self.attributes_path)[self.selected_features]

        # remove unwanted features:
        for feature in drop_features:
            if feature in self.attributes:
                self.attributes = self.attributes.drop(feature, axis=1)
                self.num_features -= 1

        self.attributes.set_index('image_id', inplace=True)
        self.attributes.replace(to_replace=-1, value=0, inplace=True)
        self.attributes['image_id'] = list(self.attributes.index)

        self.features_name = list(self.attributes.columns)[:-1]

        # load ideal partitioning:
        self.partition = pd.read_csv(self.partition_path)
        self.partition.set_index('image_id', inplace=True)

        # self.__download_data()

    def split(self, name='training', drop_zero=False):
        '''Returns the ['training', 'validation', 'test'] split of the dataset'''
        # select partition split:
        if name == 'training':
            to_drop = self.partition.where(lambda x: x != 0).dropna()
        elif name == 'validation':
            to_drop = self.partition.where(lambda x: x != 1).dropna()
        elif name == 'test':  # test
            to_drop = self.partition.where(lambda x: x != 2).dropna()
        else:
            raise ValueError('CelebA.split() => `name` must be one of [training, validation, test]')

        partition = self.partition.drop(index=to_drop.index)

        # join attributes with selected partition:
        joint = partition.join(self.attributes, how='inner').drop('partition', axis=1)

        if drop_zero is True:
            # select rows with all zeros values
            return joint.loc[(joint[self.features_name] == 1).any(axis=1)]
        elif 0 <= drop_zero <= 1:
            zero = joint.loc[(joint[self.features_name] == 0).all(axis=1)]
            zero = zero.sample(frac=drop_zero)
            return joint.drop(index=zero.index)

        return joint

    def __download_data(self):
        for fname in tqdm(self.partition.index.values):
            fpath = self.images_folder + f'/{fname}'
            fpath = _download(fpath)
        return

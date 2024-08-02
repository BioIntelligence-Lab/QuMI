from dataclasses import dataclass

import numpy as np
import pandas as pd
from skmultilearn.model_selection.iterative_stratification import (
    iterative_train_test_split,
)

# labels are 1/0
labels_8 = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
]

labels_14 = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural Thickening',
    'Hernia',
]

labels_19 = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural Thickening',
    'Pneumonia',
    'Pneumothorax',
    'Pneumoperitoneum',
    'Pneumomediastinum',
    'Subcutaneous Emphysema',
    'Tortuous Aorta',
    'Calcification of the Aorta',
]

pd.set_option('mode.copy_on_write', True)

# for convenience
labels = labels_19


def get_labels(n):
    match n:
        case 1:
            return ['Cardiomegaly']
        case 8:
            return labels_8
        case 14:
            return labels_14
        case 19:
            return labels_19
        case _:
            msg = f'{n} labels not built in, specify labels as list instead'
            raise ValueError(msg)


def load_metadata(path):
    # setting the correct dtypes and using pyarrow = faster parsing
    metadata_df = pd.read_csv(
        path,
        index_col='Image Index',
        usecols=['Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 'Image Index'],
        dtype={
            'Patient Gender': 'category',
            'View Position': 'category',
            'Patient Age': 'Int16',
            'Patient ID': 'Int16',
            'Image Index': 'string',
        },
        engine='pyarrow',
    )
    metadata_df.index = '/srv/store/Data/NIH-ChestXray-14/images/' + metadata_df.index
    return metadata_df


def load_csv(path):
    labels_df = pd.read_csv(
        path,
        index_col='id',
        dtype=dict.fromkeys(labels, bool)
        | {'subj_id': 'Int16', 'No Finding': bool, 'id': 'string'},
        engine='pyarrow',
    )
    labels_df.index = '/srv/store/Data/NIH-ChestXray-14/images/' + labels_df.index
    return labels_df


@dataclass
class Dataset:
    """Class for loading dataframes only on demand"""

    train_csv_path: str
    val_csv_path: str
    test_csv_path: str
    metadata_csv_path: str

    train_df = None
    val_df = None
    test_df = None
    metadata_df = None

    def _get_df(self, df_name):
        match df_name:
            case 'train':
                if self.train_df is None:
                    self.train_df = load_csv(self.train_csv_path)
                return self.train_df
            case 'val':
                if self.val_df is None:
                    self.val_df = load_csv(self.val_csv_path)
                return self.val_df
            case 'test':
                if self.test_df is None:
                    self.test_df = load_csv(self.test_csv_path)
                return self.test_df
            case 'metadata':
                if self.metadata_df is None:
                    self.metadata_df = load_metadata(self.metadata_csv_path)
                return self.metadata_df
            case _e:
                msg = f'{df_name} is not a valid df inside Dataset'
                raise ValueError(msg)

    # can slice the number of labels later
    # ok to split among only the labels we use. Even 19 labels is a subset of what might exist if "everything" is classified
    def get_df(self, df_name, labels=labels, frac=1, key=42):
        this_df = self._get_df(df_name)
        if frac >= 1:
            return this_df

        rng = np.random.RandomState(key)
        _, X_test, _, y_test = (
            iterative_train_test_split(  # TODO: docstring doesn't match return type
                X=np.reshape(this_df.index, (-1, 1)),  # requires a 2D array
                y=this_df[labels].to_numpy(),
                test_size=frac,
                random_state=rng,
            )
        )
        # outputs are 2D. repack into dataframe and return
        return pd.DataFrame(y_test, index=X_test[:, 0], columns=labels)


dataset = Dataset(
    train_csv_path='/srv/store/Projects/schan/quantum/PruneCXRlabels/miccai2023_nih-cxr-lt_labels_train.csv',
    val_csv_path='/srv/store/Projects/schan/quantum/PruneCXRlabels/miccai2023_nih-cxr-lt_labels_val.csv',
    test_csv_path='/srv/store/Projects/schan/quantum/PruneCXRlabels/miccai2023_nih-cxr-lt_labels_test.csv',
    metadata_csv_path='/srv/store/Data/NIH-ChestXray-14/metadata.csv',
)


if __name__ == '__main__':
    print('train')
    print(len(dataset.get_df('train', labels_8, frac=1)))
    print('val')
    print(len(dataset.get_df('val', labels_8, frac=1)))
    import jax.random as jrand
    for frac in [0.5]: # [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75]:
        train_lens = []
        val_lens = []
        for seed in list(range(100, 106)):
            key = jrand.PRNGKey(seed)
            # use the same keys as in training, in case fractions don't divide evenly
            key, model_key = jrand.split(key, num=2)
            key, train_key, val_key = jrand.split(key, num=3)
            train_lens.append(len(dataset.get_df('train', labels_8, frac=frac, key=key)))
            val_lens.append(len(dataset.get_df('val', labels_8, frac=frac, key=key)))
        print(f'train {frac} {seed}')
        print(train_lens)
        print(f'val {frac} {seed}')
        print(val_lens)

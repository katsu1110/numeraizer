import numpy as np
import pandas as pd
import pathlib
import json
from tqdm.auto import tqdm

import numerapi

class NumeraiDataHandler:
    """
    data handler class via Numerapi
    """
    train_path = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
    tournament_path = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'
    
    def __init__(self, config_dir: str, output_dir: str='./'):
        self.config_dir = config_dir
        self.output_dir = output_dir

    @classmethod
    def fetch_data(cls, data_type='train'):
        # fetch data
        if data_type.lower() == 'train':
            url = cls.train_path
        elif data_type.lower() in ['test', 'live']:
            url = cls.tournament_path
        df = pd.read_csv(url)

        # cast to uint8 to reduce memory demand
        feature_cols = df.columns[df.columns.str.startswith('feature')]
        mapping = {0.0 : 0, 0.25 : 1, 0.5 : 2, 0.75 : 3, 1.0 : 4}
        for c in feature_cols:
            df[c] = df[c].map(mapping).astype(np.uint8)
        
        # also cast era to int
        def get_int(x):
            try:
                return int(x[3:])
            except:
                return 1000
        df["era"] = df["era"].apply(get_int)
        return df

    def load_train(self):
        raise NotImplementedError()

    def load_tournament(self):
        raise NotImplementedError()

    def train_valid_test_split(self):
        # load
        train = self.load_train()
        tournament = self.load_tournament()
        
        # extract valid
        valid = tournament.query('data_type == "validation"').reset_index(drop=True)
        
        # remove data type
        train.drop(columns=['data_type'], inplace=True)
        valid.drop(columns=['data_type'], inplace=True)
        tournament.drop(columns=['data_type'], inplace=True)
        return train, valid, tournament

    def api_setup(self):
        with open(self.config_dir) as f:
            numerai_keys = json.load(f)
        public_id = numerai_keys['public_id']
        secret_key = numerai_keys['secret_key']
        return public_id, secret_key

    def get_napi(self):
        if self.config_dir is not None:
            print('Authentification...')
            public_id, secret_key = self.api_setup()
            napi = numerapi.NumerAPI(public_id, secret_key)
        else:
            print('Using Numerapi without Authentification...')
            napi = numerapi.NumerAPI()
        return napi

    @staticmethod
    def normalize(v, newmin, newmax):
        a = (newmax - newmin)/(max(v) - min(v))
        b = newmax - a*max(v)

        return [a*i + b for i in v]

    def submit(self, tournament : pd.DataFrame, pred : np.ndarray, lot_name: str='NYAAA'):
        """
        submit your Numerai Tournament predictions to Numerai
        """
        prediction = 'prediction'
        predictions_df = tournament["id"].to_frame()
        predictions_df[prediction] = pred

        # 0 - 1
        pred = self.normalize(predictions_df[prediction].values, 0.2, 0.8)
        predictions_df[prediction] = np.array(pred)

        # get lot name - id
        napi = self.get_napi()
        lot_ids = napi.get_models()
        model_id = lot_ids[lot_name]
        
        # save
        save_path = f"{self.output_dir}/predictions_{lot_name}_{model_id}.csv"
        predictions_df.to_csv(pathlib.Path(save_path), index=False)
        
        # Upload your predictions using API
        submission_id = napi.upload_predictions(pathlib.Path(save_path), model_id=model_id)
        print(f'submitted to {lot_name}: {model_id}!')
        
        return predictions_df
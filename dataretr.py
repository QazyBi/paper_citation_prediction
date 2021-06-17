from typing import List
from tqdm import tqdm
import pandas as pd
import requests
import os
import gzip


def download_n_store(filenames: List[str], link: str, out_dir: str):
    """

    """
    for file in tqdm(filenames, desc="downloading files "):
        # make a request to the s3 service
        response = requests.get(link + file)
        # path to the file
        path = os.path.join(out_dir, file)
        # write contents of the request to the file
        with open(path, 'wb+') as f:
            f.write(response.content)


    
def make_df(filenames: List[str], out_dir: str) -> pd.DataFrame:
    """

    """
    corpus = pd.DataFrame()

    for file in tqdm(filenames, desc="building dataframe"):
        # path to the file
        path = os.path.join(out_dir, file)

        # unzip this file
        with gzip.open(path, 'rb') as f:    
            data = f.read()        

        # make dataframe out of the file    
        df = pd.read_json(data, orient='records', lines=True)
        # join new dataframe with corpus dataframe 
        corpus = pd.concat([corpus, df])
        # delete temporary dataframe
        del df  

    corpus.set_index('id', inplace=True)
    return corpus


def download_make_df(filenames: List, link: str, out_dir: str) -> pd.DataFrame:
    """

    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if set(filenames) != set(os.listdir(out_dir)):
        download_n_store(filenames=filenames, link=link, out_dir=out_dir)

    
    corpus = make_df(filenames, out_dir)

    return corpus


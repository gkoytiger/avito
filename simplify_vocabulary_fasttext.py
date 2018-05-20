import os
from gensim.models.fasttext import FastText
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.decomposition import TruncatedSVD
#functions from https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


DATA_FOLDER = '/home/greg/.kaggle/competitions/avito-demand-prediction/'
INPUT_DB = 'baseline_features.sqlite'
TEXT_DB = 'text_features.sqlite'
OUTPUT_DB = 'engineered_features.sqlite'
WORD_VECTORS = os.path.join(DATA_FOLDER, 'wiki.ru')

text_db = create_engine('sqlite:///'+os.path.join(DATA_FOLDER, 'derived', TEXT_DB))

ft = FastText.load_fasttext_format(WORD_VECTORS)
vocabulary = pd.read_sql_table('clean_vocabulary', text_db)

word_vecs = ft[vocabulary['word'].values]
word_vecs = remove_pc(word_vecs)
word_df = pd.DataFrame(data=word_vecs, index=vocabulary['word'])

word_df.to_sql('word_vecs', text_db)
import numpy as np
import pandas as pd
import requests
from info import *
import json
from collections import Counter
from scholarmetrics import hindex
import urllib
from tqdm import tqdm
from typing import List, Dict
import tqdm.notebook as tq


class AuthorPrep:
    """This is a class for preprocessing author columns 
    
    Attributes
    ----------
    max_id: int
        maximal author id
    author_ids: dict
        mapping between authors and ids
    authors_without_id: list
        list of authors without id

    Methods
    -------
    get_max_id(corpus)
        Finds max id for authors
    
    generate_ids()
        Generating new ids for authors without them

    get_author_ids()
        Finds ids of authors for a given paper
    """
    def __init__(self):
        self.max_id = 0
        self.author_ids = {}
        self.authors_without_id = []
        
    def get_max_id(self, corpus: pd.DataFrame):
        """Function for finding maximal author id through iterating over all ids


            Parameters
            ----------
            corpus: pd.DataFrame
                corpus containing papers meta-information
        """
        for i, row in tqdm(enumerate(corpus.authors.to_numpy())):
            if row is np.nan:
                continue
            # iterate over each paper authors
            for author in row:
                id = author['ids']
                name = author['name']
                # if no id then append it to the list of authors without id
                if id == []:
                    self.authors_without_id.append((name, i))
                else:
                    int_id = int(id[0])
                    # if author is in the author_ids dictionary then add new element
                    if name in self.author_ids:
                        self.author_ids[name].append((int_id, i))
                    else:
                        # if author is not in the author_ids dictionary then create list of ids
                        self.author_ids[name] = [(int_id, i)]
                    # update max_id variable if new biggest id found
                    if int_id > self.max_id:
                        self.max_id = int_id
                        
    def generate_ids(self):
        """Generate ids for authors without them 

        """
        new_id = self.max_id
        for author in self.authors_without_id:
            new_id += 1

            if author[0] in self.author_ids:
                self.author_ids[author[0]].append((new_id, author[1]))
            else:
                self.author_ids[author[0]] = [(new_id, author[1])]
        
        self.authors_without_id = []        

    def get_author_ids(self, row: pd.Series) -> List:
        """Provide author ids for a given paper

            Parameters
            ----------
            row: pd.Series
                Series with paper meta-information

            Returns
            -------
                res: List - list of authors ids
        """
        if row is np.nan:
            return np.nan

        res = []
        for i, author in enumerate(row):
            id_list = author['ids']
            name = author['name']

            if len(id_list) == 0:
                ids = self.author_ids[name]

                for id, index in ids:
                    if index == i:
                        res.append(id)
            else:
                res.append(id_list[0])

        return res


def get_paper(pid: str):
    """Function retrieving paper meta-information

        Parameters
        ----------
            pid: str - paper identifier in the Semantic Scholar database

        Returns
        -------
            res: pd.Series - paper meta-information
    """
    url = PAPER_API + pid

    response = requests.get(url, proxies=urllib.request.getproxies()) 
    res = json.loads(response.text)
    return pd.Series(res)    


# NEW FEATURE: hindex of the author with highest hindex value
class CitationsPrep:
    def is_invalid_paper(self, corpus: pd.DataFrame, pid: str, p_year: int) -> bool:
        """Test whether paper is invalid
        """
        try:
            cited_paper = corpus.loc[pid]
        except KeyError:
            # cited_paper = get_paper(pid)  # it takes a lot of time better to switch on parallel computations
            return False

        try:
            cited_year = cited_paper.year
    
            if cited_year - p_year >= 5:
                return True
        except:
            pass

        return False
    
    def count(self, corpus: pd.DataFrame, cite_col='inCitations'):
        """Calculates impact factor of the authors

        """
        # paper citation counter
        p_cite_count = Counter()
        # author citations counter 
        a_cite_count = Counter()

        # pid - paper id
        for pid in tq.tqdm(corpus.index, mininterval=15):
            #
            cited_papers = corpus.loc[pid, cite_col]
            if cited_papers is np.nan:
                continue

            p_year = corpus.loc[pid].year        
            #
            for cited_pid in cited_papers:
                #
                if self.is_invalid_paper(corpus, cited_pid, p_year):
                    continue

                p_cite_count[pid] += 1


                ## Authors citations ##
                
                authors = corpus.loc[pid].authors 
                if  authors is np.nan:
                    continue

                # update authors citation counters
                for id in authors:
                    if id not in a_cite_count: 
                        a_cite_count[id] = Counter()

                    a_cite_count[id][pid] += 1

        return p_cite_count, a_cite_count

    def calc_scholarmetrics(self, corpus, p_cite_count, a_cite_count):
        """Calculate papers and authors hindices
        """
        max_h_values = Counter()

        # max h-index
        for pid in tq.tqdm(corpus.index):
            if corpus.loc[pid].authors is np.nan:
                continue

            h_indexes = []

            for id in corpus.loc[pid].authors:
                if id in a_cite_count:
                    citations = list(a_cite_count[id].values())
                    h_index = hindex(citations)
                else:
                    h_index = 0
                
                h_indexes.append(h_index)

            if len(h_indexes) == 0:
                max_h = 0
            else:
                primary_id = np.argmax(h_indexes)
                max_h  = np.max(h_indexes)

            max_h_values[pid] = max_h
            
        return max_h_values

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 1 12:00:00 2025

@author: tadahaya
"""
import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
from tqdm.auto import tqdm

class Checker:
    def __init__(
            self,
            terms:list=["Nasopharyngitis", "Rhinitis", "Diverticulitis", "Otitis media chronic", "Gastroenteritis"],
            outdir:str=""
            ):
        self.ddf = None
        self.workdir = None
        self.terms = {v.lower() for v in terms}
        self.outdir = outdir
        # self._black_list = ["+", "and"] # Inactive terms
        self._black_list = []

    def load_data(self, url, verbose:bool=False):
        self.ddf = dd.read_csv(url, dtype="object")
        self.workdir = os.path.dirname(url)
        if verbose:
            print("> loaded data")
            print(self.ddf.head(5))

    def select_nct(self, name_drug):
        """
        extracts nct_id_x, title, description from ddf where

        """
        # select nct_id_x and title
        ddf = self.ddf[
            ["nct_id_x", "title"]
            ]
        # select titles that contain the drug name
        ddf = ddf[
            ddf["title"].apply(lambda x: name_drug in x.lower() if type(x) == str else False,
            meta=("title", "bool"))
        ]
        # obtain nct_id_x related to the drug
        nct = ddf[["nct_id_x"]].drop_duplicates(subset=["nct_id_x"], keep="first").compute(scheduler="processes")
        nct = set(list(nct["nct_id_x"]))
        # restart filtering
        ddf = self.ddf[
            ["nct_id_x", "adverse_event_term", "title", "description"]
            ]
        ddf = ddf[ddf["nct_id_x"].isin(nct)]
        # select event_term
        ddf = ddf[
            ddf["adverse_event_term"].apply(lambda x: x.lower() in self.terms if type(x) == str else False,
            meta=("adverse_event_term", "bool"))
        ]
        # remove the terms in the blacklist
        for b in self._black_list:
            ddf = ddf[
                ddf["title"].apply(lambda x: b.lower() not in x.lower() if type(x) == str else False,
                meta=("title", "bool"))
                ]
        # filter nct_id_x, title, and description
        ddf = ddf[["nct_id_x", "title", "description"]]
        # drop duplicates
        ddf = ddf.drop_duplicates(subset=["nct_id_x", "title"], keep="first")
        # prepare df
        df = ddf.compute(scheduler='processes')
        # prepare columns for check
        df["treated"] = np.nan
        # export
        df = df.sort_values("nct_id_x")
        df = df.reset_index(drop=True)
        if self.outdir != "":
            df.to_csv(self.outdir + f"/checker_{name_drug}.csv")
        return df
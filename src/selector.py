# -*- coding: utf-8 -*-
"""
Created on Fri Aug 1 12:00:00 2025

@author: tadahaya
"""
import os
import pandas as pd
import dask.dataframe as dd
from tqdm.auto import tqdm

class Selector:
    def __init__(
            self, url_ddf:str="", url_meta:str="", outdir:str="", drug:str="",
            terms:list=["Nasopharyngitis", "Rhinitis", "Diverticulitis", "Otitis media chronic", "Gastroenteritis"],
            drop_pseudo_placebo:bool=False, drop_pseudo_treated:bool=False
            ):
        self.ddf = dd.read_csv(url_ddf, dtype="object")
        self.preprocessed = None
        self.processed = None
        self.outdir = outdir
        self.terms = [v.lower() for v in terms]
        self.drug = drug
        self.drop_pseudo_placebo = drop_pseudo_placebo
        self.drop_pseudo_treated = drop_pseudo_treated
        # prepare meta
        meta = pd.read_csv(url_meta, index_col=0)
        self.meta = meta.dropna(subset=["treated"]).reset_index(drop=True) # remove rows with NaN in 'treated'
        self.nct = set(list(self.meta["nct_id_x"])) # obtain unique nct_id_x
        self.title = set(list(self.meta["title"])) # obtain unique titles
        # prepare meta_dict
        self.meta_dict = dict()
        for n in self.nct:
            tmp = self.meta[self.meta["nct_id_x"] == n]
            titles = list(tmp["title"])
            treats = list(tmp["treated"])
            for tit, tre in zip(titles, treats):
                self.meta_dict[(n, tit)] = tre


    def preprocess(self):
        """
        extract necessary data referring to meta

        """
        # filter nct_id_x, subjects_affected, subjects_at_risk, adverse_event_term, and title
        ddf = self.ddf[
            ["nct_id_x", "subjects_affected", "subjects_at_risk", "adverse_event_term", "title"]
            ]
        # filter nct_id_x
        ddf = ddf[ddf["nct_id_x"].isin(self.nct)]
        # filter title
        ddf = ddf[ddf["title"].isin(self.title)]
        # fix type
        ddf["subjects_affected"] = ddf["subjects_affected"].map_partitions(self._to_numeric)
        ddf["subjects_at_risk"] = ddf["subjects_at_risk"].map_partitions(self._to_numeric)
        # remove rows with NaN in 'subjects_at_risk'
        ddf = ddf.dropna(subset=["subjects_at_risk"])
        # subjects_at_risk > 0
        ddf = ddf[ddf["subjects_at_risk"] > 0]
        # prepare df
        df = ddf.compute(scheduler='processes')
        # add treated column
        df["treated"] = df.apply(lambda x: self.meta_dict[(x["nct_id_x"], x["title"])], axis=1)
        # convert adverse event term to lowercase
        df["adverse_event_term"] = df["adverse_event_term"].map(lambda x: x.lower())
        # export
        df = df.sort_values("nct_id_x")
        df = df.reset_index(drop=True)
        self.preprocessed = df
        return df


    def select(self, latent_val:float=0.1):
        """
        organize the data into a DataFrame with the following columns:
        nct_id_x, subjects_affected, subjects_at_risk, adverse_event_term, title, treated
        loop over event terms
        loop over nct
        check if there is a pair of treated and placebo, and process accordingly

        """
        # loop over event terms
        result = []
        for term in self.terms:
            df_term = self.preprocessed[self.preprocessed["adverse_event_term"] == term]
            if df_term.shape[0] > 0:
                tmp_nct = list(set(list(df_term["nct_id_x"])))
                # loop over nct
                res_nct = []
                for n in tmp_nct:
                    df_nct = df_term[df_term["nct_id_x"] == n]
                    treated = set(list(df_nct["treated"]))
                    res_nct.append(df_nct)
                    assert len(treated) <= 2
                    if len(treated) == 2: # dual groups exist
                        pass
                    elif (len(treated) == 1) & (list(treated)[0] == 0): # only Placebo
                        if self.drop_pseudo_treated:
                            pass
                        else:
                            # add pseudo-treated and average at_risk
                            ave = df_nct["subjects_at_risk"].values.mean()
                            tmp = pd.DataFrame(
                                [[n, latent_val, ave, term, "PSEUDO-TREATED", 1],],
                                columns=["nct_id_x", "subjects_affected", "subjects_at_risk", "adverse_event_term", "title", "treated"]
                                )
                            res_nct.append(tmp)
                    elif (len(treated) == 1) & (list(treated)[0] == 1): # treatedのみ
                        if self.drop_pseudo_placebo:
                            pass
                        else:
                            # add pseudo-placebo and average at_risk
                            ave = df_nct["subjects_at_risk"].values.mean()
                            tmp = pd.DataFrame(
                                [[n, latent_val, ave, term, "PSEUDO-PLACEBO", 0],],
                                columns=["nct_id_x", "subjects_affected", "subjects_at_risk", "adverse_event_term", "title", "treated"]
                                )
                            res_nct.append(tmp)
                    else:
                        print(tmp_nct)
                        raise ValueError("treatedがおかしい")
                res_nct = pd.concat(res_nct, axis=0, join="inner")
                # export
                res_nct = res_nct.sort_values("nct_id_x")
                res_nct = res_nct.reset_index(drop=True)
            else:
                res_nct = pd.DataFrame(
                    columns=["nct_id_x", "subjects_affected", "subjects_at_risk", "adverse_event_term", "title", "treated"]
                    )

            result.append(res_nct)

        df = pd.concat(result, axis=0, join="inner")
        # implementation
        df["subjects_affected"] = df["subjects_affected"].astype(float)
        df = df.fillna({"subjects_affected": latent_val})
        # calculate ratio
        df["ratio"] = df["subjects_affected"] / df["subjects_at_risk"]
        # export
        df = df.reset_index(drop=True)
        self.processed = df
        if self.outdir != "":
            df.to_csv(self.outdir + f"/selected_{self.drug}.csv")
        return df

    def _to_numeric(self, df):
        df = pd.to_numeric(df, errors="coerce")
        return df
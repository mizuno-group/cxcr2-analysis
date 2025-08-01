# -*- coding: utf-8 -*-
"""
Created on Fri Aug 1 12:00:00 2025

@author: tadahaya
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RiskCalculator:
    def __init__(
            self, input_url:str, drug:str="", outdir:str="",
            ):
        self.input = pd.read_csv(input_url, index_col=0)
        self.preprocessed = None
        self.boot_result = None
        self.each_result = None
        self.result = None
        self.drug = drug
        self.outdir = outdir
        # obtain event terms
        self.terms = list(set(list(self.input["adverse_event_term"])))
        # obtain nct_id_x
        self.nct = list(set(list(self.input["nct_id_x"])))


    def preprocess(self):
        """
        convert input data to a dictionary format like:
        {
            term:{
                    nct_id_x:{
                                placebo:[(subjects_affected, subjects_at_risk), ...],
                                treated:[(subjects_affected, subjects_at_risk), ...],
                    }
                },
            }
        }

        """
        result = dict()
        for term in self.terms:
            df_term = self.input[self.input["adverse_event_term"] == term]
            result[term] = dict()
            for n in self.nct:
                df_nct = df_term[df_term["nct_id_x"] == n]
                if df_nct.shape[0] > 0:
                    placebo = df_nct[df_nct["treated"] == 0]
                    treated = df_nct[df_nct["treated"] == 1]
                    result[term][n] = {
                        "placebo":list(zip(list(placebo["subjects_affected"]), list(placebo["subjects_at_risk"]))),
                        "treated":list(zip(list(treated["subjects_affected"]), list(treated["subjects_at_risk"])))
                        }
            if len(result[term]) == 0:
                del result[term]
        self.preprocessed = result


    def main(self, n_samples:int=5000, log:bool=True):
        """
        1. calculate risk ratio
        2. export results

        """
        # 0. preprocess data
        self.preprocess()
        # 1. calculate risk ratio
        result = self.calc(n_samples, log)
        # 2. export results
        self.export_each(log)
        df = []
        for k, v in result.items():
            final_stat, final_samples = v
            final_mean, lower_bound, upper_bound = final_stat
            value = [self.drug, k, final_mean, lower_bound, upper_bound]
            df.append(value)
            # visualize
            plt.figure()
            plt.hist(final_samples, bins=100, color="navy", alpha=0.5, density=True)
            plt.title(f"{self.drug} / {k}")
            plt.tight_layout()
            if len(self.outdir) > 0:
                plt.savefig(self.outdir + f"/hist_{self.drug}_{k}.png")
            plt.show()
        if log:
            col = ["name_drug", "adverse_event_term", "log_rr", "lower_ci", "upper_ci"]
        else:
            col = ["name_drug", "adverse_event_term", "rr", "lower_ci", "upper_ci"]
        df = pd.DataFrame(df, columns=col)
        if len(self.outdir) > 0:
            df.to_csv(self.outdir + f"/res_{self.drug}.csv")
        self.result = df
        return df


    def export_each(self, log=True):
        df = []
        for k, v in self.each_result.items():
            for n, val in v.items():
                stat, ci = val
                lower_bound, upper_bound = ci
                value = [self.drug, k, n, stat, lower_bound, upper_bound]
                df.append(value)
        if log:
            col = ["name_drug", "adverse_event_term", "nct_id_x", "log_rr", "lower_ci", "upper_ci"]
        else:
            col = ["name_drug", "adverse_event_term", "nct_id_x", "rr", "lower_ci", "upper_ci"]
        df = pd.DataFrame(df, columns=col)
        if len(self.outdir) > 0:
            df.to_csv(self.outdir + f"/res_detail_{self.drug}.csv")


    def calc(self, n_samples:int=5000, log:bool=True):
        """ calculation """
        # loop over event terms
        result = dict()
        result_each = dict()
        for term in self.terms:
            # bootstrap samples in each nct
            inner_samples = []
            result_each[term] = dict()
            for n in self.nct:
                try:
                    placebo_data = self.preprocessed[term][n]["placebo"]
                    treated_data = self.preprocessed[term][n]["treated"]
                    stat, ci = self.bootstrap_risk_ratio(
                        placebo_data, treated_data, n_samples, log
                        )
                    result_each[term][n] = (stat, ci)
                    inner_samples.append(stat) # ignore ci at this point
                except KeyError:
                    pass
            n = len(inner_samples) # the number of nct
            if n > 0:
                inner_samples = np.array(inner_samples)
                # bootstrap samples
                final_boot = []
                for _ in range(n_samples):
                    idx = np.random.choice(n, size=n, replace=True)
                    resampled = inner_samples[idx]
                    mean = np.average(resampled)
                    final_boot.append(mean)
                # calculate bootstrap confidence interval
                final_boot = np.array(final_boot)
                final_mean = np.mean(final_boot)
                lower_bound, upper_bound = np.percentile(final_boot, (2.5, 97.5))
                final_stat = (final_mean, lower_bound, upper_bound)
                # save results
                result[term] = (final_stat, final_boot)
            else:
                del result_each[term]
        self.boot_result = result
        self.each_result = result_each
        return result


    def weighted_mean(self, data):
        """ calculate weighted mean """
        weights = [n for _, n in data]
        proportions = [k / n for k, n in data]
        return np.average(proportions, weights=weights)


    def bootstrap_risk_ratio(self, placebo_data, treated_data, n_bootstrap=1000, log=True, eps=1e-6):
        """ Calculate the risk ratio using bootstrap sampling."""
        # prepare bootstrap samples
        rr_samples = []
        for _ in range(n_bootstrap):
            # resample Placebo and Treated
            placebo_sample = [placebo_data[i] for i in np.random.randint(0, len(placebo_data), len(placebo_data))]
            treated_sample = [treated_data[i] for i in np.random.randint(0, len(treated_data), len(treated_data))]
            # calculate integrated risk with weighted average
            placebo_risk = self.weighted_mean(placebo_sample) + eps
            treated_risk = self.weighted_mean(treated_sample) + eps
            # log
            if log:
                placebo_risk = np.log(placebo_risk)
                treated_risk = np.log(treated_risk)
                rr_samples.append(treated_risk - placebo_risk)
            else:
                rr_samples.append(treated_risk / placebo_risk)
            # calculate confidence interval
        lower_ci = np.percentile(rr_samples, 2.5)
        upper_ci = np.percentile(rr_samples, 97.5)
        return np.mean(rr_samples), (lower_ci, upper_ci)
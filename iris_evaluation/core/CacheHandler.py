import os
import numpy as np
import polars as pl

class CacheHandler:
    def __init__(self, target, self_comparison):
        self.__path = f'{target}/.cache.tsv'
        self.__self_comparison = self_comparison #boolean indicating whether or not comparisons are intradataset
        self.clear()
        
    def new_line(self, comparison, result):
        with open(self.__path, 'a') as file:
            file.write(f'{comparison}\t{result}\n')

    def clear(self):
        if os.path.exists(self.__path):
            os.remove(self.__path)
    
    def save(self):
        df = self.__import()
        path = self.__path.replace('/.cache.tsv', '/cache.parquet')
        df.write_parquet(path)

    def __import(self):
        df = pl.read_csv(self.__path, has_header=False, separator='\t', new_columns=['comparison', 'HD'])
        df = df.unique(subset=["comparison"])
        df = df.with_columns(pl.col("comparison").str.split_exact("|", 1).struct.rename_fields(["img1", "img2"]).alias("fields")).unnest("fields")
        if self.__self_comparison:
            df = df.filter(pl.col("img1") != pl.col("img2"))

        return df

    def __get_hamming_distances(self):
        df = self.__import()
        hamming_distances = df["HD"].to_numpy()

        return hamming_distances

    def __get_stats(self, get_all = False):
        hamming_distances = self.__get_hamming_distances()
        hamming_stats = {
            "mean": np.mean(hamming_distances),
            "stdev": np.std(hamming_distances),
        }
        
        return hamming_stats
    
    def __get_all(self): #probably not the most efficient way to do this, but it is what it is.
        df = self.__import()
        hamming_distances = df["HD"].to_numpy()
        hamming_stats = {
            "mean": np.mean(hamming_distances),
            "stdev": np.std(hamming_distances),
        }

        all_data = tuple(df, hamming_distances, hamming_stats)

        return all_data


    def __get_path(self):
        return self.__path
    
    def __get_mode(self):
        if self.__self_comparison:
            return "self_comparison"
        else:
            return "normal"

    def __get_all(self):
        df = self.__import()
    
    #sets properties
    path = property(fget = __get_path)
    mode = property(fget = __get_mode)
    cache = property(fget = __import)
    hamming_distances = property(fget = __get_hamming_distances)
    hamming_stats = property(fget = __get_stats)
    all_stats = property(fget = __get_all)
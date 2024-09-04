import os
import numpy as np
import polars as pl

class CacheHandler:
    def __init__(self, target, self_comparison=False):
        self.path = f'{target}/.cache.tsv'
        self.self_comparison = self_comparison #boolean indicating whether or not comparisons are intradataset
        self.clear()
        
    def new_line(self, comparison, result):
        with open(self.path, 'a') as file:
            file.write(f'{comparison}\t{result}\n')

    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)
    
class PairwiseCache(CacheHandler):
    def save(self):
        df = self.__import()
        path = self.path.replace('/.cache.tsv', '/cache.parquet')
        df.write_parquet(path)

    def __import(self):
        df = pl.read_csv(self.path, has_header=False, separator='\t', new_columns=['comparison', 'HD'])
        df = df.unique(subset=["comparison"])
        df = df.with_columns(pl.col("comparison").str.split_exact("|", 1).struct.rename_fields(["img1", "img2"]).alias("fields")).unnest("fields")
        if self.self_comparison:
            df = df.filter(pl.col("img1") != pl.col("img2"))

        return df

    cache = property(fget = __import)

    def __get_hamming_distances(self):
        df = self.__import()
        hamming_distances = df["HD"].to_numpy()

        return hamming_distances
    
    hamming_distances = property(fget = __get_hamming_distances)

    def __get_stats(self):
        hamming_distances = self.__get_hamming_distances()
        hamming_stats = {
            "mean": np.mean(hamming_distances),
            "stdev": np.std(hamming_distances),
        }
        
        return hamming_stats
    
    hamming_stats = property(fget = __get_stats)
    
    def __get_all(self): #probably not the most efficient way to do this, but it is what it is.
        df = self.__import()
        hamming_distances = df["HD"].to_numpy()
        hamming_stats = {
            "mean": np.mean(hamming_distances),
            "stdev": np.std(hamming_distances),
        }

        all_data = tuple(df, hamming_distances, hamming_stats)

        return all_data

    all_data = property(fget = __get_all)

class LinearCache(CacheHandler):
    def save(self):
        df = self.__import()
        path = self.path.replace('/.cache.tsv', '/cache.parquet')
        df.write_parquet(path)

    def __import(self):
        df = pl.read_csv(self.path, has_header=False, separator='\t', new_columns=['comparison', 'HD'])
        df = df.unique(subset=["comparison"])
        df = df.with_columns(pl.col("comparison").str.split_exact("|", 1).struct.rename_fields(["img1", "img2"]).alias("fields")).unnest("fields")
        if self.self_comparison:
            df = df.filter(pl.col("img1") != pl.col("img2"))

        return df
    
    def __import(self):
        df = pl.read_csv(self.path, has_header=False, separator='\t', new_columns=['comparison', 'HD'])
        df = df.with_columns(pl.col("comparison").str.split_exact("|", 1).struct.rename_fields(["image", "condition"]).alias("fields")).unnest("fields")
        df = df.with_columns(pl.col("condition").str.split_exact("_", 1).struct.rename_fields(["rotation", "mask"]).alias("fields")).unnest("fields")
        df = df.select(["image", "condition", "rotation", "mask", "HD", "comparison"])
        df = df.drop("comparison")


        return df

    cache = property(fget = __import)
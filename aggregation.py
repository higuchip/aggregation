import numpy as np
import pandas as pd
import scipy.stats as stats


class Aggregation:
    """Cálculo de padrões de agregação

    Argumento:
        file: arquivo de dados no formato csv

    Retorno:

        pandas series e dataframe com os resultados da análise de agregação
        determinados pelo método de McGinnies, Racker-Brischle e Morisita

    """

    def __init__(self, file):
        self._file = file
        self._df = pd.read_csv(file, sep=";", decimal=",", encoding="latin-1")
        self._ni = self._df.Especie.value_counts()
        self._ut = self._df["Parcela"].nunique()
        self._Di = self._ni / self._ut
        self._ui = self._df.groupby("Especie")["Parcela"].nunique()
        self._fri = self._ui / self._ut
        self._di = -np.log(1 - self._fri)

    def __str__(self) -> str:
        return f"Análise de agregação"

    def mcginnies(self):
        return self._Di / self._di

    def racker_brischle(self):
        return (self._Di - self._di) / self._di**2

    def morisita(self):
        self._table = pd.crosstab(self._df["Parcela"], self._df["Especie"])
        self._table_squared = self._table**2
        self._table_squared_sum = self._table_squared.sum(axis=0)
        self._nplots = self._table.shape[0]
        _mori = self._nplots * (
            (self._table_squared_sum - self._ni) / (self._ni * (self._ni - 1))
        )
        _x2Sup = stats.chi2.ppf(1 - 0.05, df=self._nplots - 1)
        _x2Inf = stats.chi2.ppf(0.05, df=self._nplots - 1)
        _limAgr = (_x2Sup - self._nplots + self._ni) / (self._ni - 1)
        _limUni = (_x2Inf - self._nplots + self._ni) / (self._ni - 1)
        _morisitaIndex = pd.concat([_mori, _limAgr, _limUni], axis=1)
        _morisitaIndex.columns = ["Morisita Index", "LimAgr", "LimUni"]
        return _morisitaIndex

import numpy as np
from scipy.signal import savgol_filter

from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import FLOAT_DTYPES


class MultipleScatterCorrection(TransformerMixin, BaseEstimator):

    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=True, accept_sparse='csr', copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        X = MSC(X)
        return X

    def _more_tags(self):
        return {'allow_nan': True}


class RobustNormalVariate(TransformerMixin, BaseEstimator):

    def __init__(self, *, iqr1=75, iqr2=25, copy=True):
        self.copy = copy

        self.iqr1 = iqr1
        self.iqr2 = iqr2

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=True, accept_sparse='csr', copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        X = RNV(X)
        return X

    def _more_tags(self):
        return {'allow_nan': True}


class StandardNormalVariate(TransformerMixin, BaseEstimator):

    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        return self

    def transform(self, X, copy=None):
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=True, accept_sparse='csr', copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        X = SNV(X)
        return X

    def _more_tags(self):
        return {'allow_nan': True}


class SavitzkyGolay(TransformerMixin, BaseEstimator):

    def __init__(self, *, filter_win=9, poly_order=3, deriv_order=2, delta=1.0, copy=True):
        self.copy = copy
        self.filter_win = filter_win
        self.poly_order = poly_order
        self.deriv_order = deriv_order
        self.delta = delta

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        # Make sure filter window length is odd
        filter_win = self.filter_win
        if self.filter_win % 2 == 0:
            filter_win += 1

        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=True, accept_sparse='csr', copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        X = savgol_filter(X, window_length=filter_win, polyorder=self.poly_order, deriv=self.deriv_order, delta=self.delta)
        return X

    def _more_tags(self):
        return {'allow_nan': True}

    
def MSC(X):
    "Multiplicative Scatter Correction."
    Xmean = np.mean(X, axis=0)
    Xmsc = np.zeros_like(X)
    for i in range(X.shape[0]):
        a, b = np.polyfit(Xmean, X[i, :], deg=1)
        Xmsc[i, :] = (X[i, :]-b)/a
    return(Xmsc)


def RNV(X):
    " Robust Normal Variate transformation"
    iqr = [75, 25]
    Xt = X.T
    Xrnv = (Xt - np.median(Xt, axis=0))/np.subtract(*np.percentile(Xt, iqr, axis=0))
    Xrnv = Xrnv.T
    return(Xrnv)


def SNV(X):
    "Standard Normal Variate"
    Xt = X.T
    Xsnv2 = (Xt - np.mean(Xt, axis=0))/np.std(Xt, axis=0)
    Xsnv = Xsnv2.T
    return(Xsnv)
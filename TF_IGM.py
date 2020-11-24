from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import numpy as np

class TfigmTransformer(TfidfTransformer):

    def __init__(self, classes, landa=7):
        self.classes =classes
        self.landa =landa
        super(TfigmTransformer, self).__init__()

    def f1it_transform(self, X, y, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        return self.fit(X, y).transform(X)

    def fit(self, X, y):
        classes = self.classes
        y =np.array(y)
        doc_class_index =[np.where(np.array(y) == classItem) for classItem in classes]

        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = [self._document_frequency(X[classIndexes], len(classes)) for classIndexes in doc_class_index]
            # sort and transpose the array to make it term-frequency-class array
            df= np.sort(np.array(df).transpose())
            # perform idf smoothing if required
            df = df.reshape(df.shape[0], df.shape[2])
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            r = classes[::-1]+1

            f_kr = df[:,-1]
            # self._idf_diag = np.divide(f_kr, np.dot(df, r))
            igm = np.divide(f_kr, np.dot(df, r))
            #
            # idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(igm, diags=0, m=n_features,
                                        n=n_features, format='csr')

        return self

    def transform(self, X, copy=True):
        landa = self.landa
        """Transform a count matrix to a tf or tf-idf representation
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            w_igm =  np.multiply(landa, self._idf_diag)
            w_igm.data = w_igm.data +1
            X= X * w_igm
        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X


    def _document_frequency(sefl, X, classNumber):
        """Count the number of non-zero values for each feature in sparse X."""
        result = np.zeros(X.shape[1])
        if sp.isspmatrix_csr(X):
           return np.sum(X, axis=0)
        # else:
        #     return np.diff(sp.csc_matrix(X, copy=False).indptr)






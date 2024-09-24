import sys
import os
from tqdm import tqdm
from colorama import Fore
import pickle

import multiprocessing as mp
import threading as th

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram

import matplotlib.pyplot as plt

from sklearn.preprocessing import (Normalizer,
                                   StandardScaler, MinMaxScaler, MaxAbsScaler,
                                   RobustScaler, QuantileTransformer, PowerTransformer)

from sklearn.linear_model import (SGDClassifier, SGDOneClassSVM, RidgeClassifier, RidgeClassifierCV,
                                  PassiveAggressiveClassifier)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
                                  OrthogonalMatchingPursuit,
                                  BayesianRidge, ARDRegression, SGDRegressor, RANSACRegressor, GammaRegressor,
                                  PoissonRegressor, HuberRegressor,
                                  TweedieRegressor, LogisticRegression, QuantileRegressor, TheilSenRegressor)
from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor,
                               RadiusNeighborsClassifier, RadiusNeighborsRegressor)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.tree import plot_tree

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor,
                              StackingRegressor, VotingRegressor)

from sklearn.metrics import (mean_absolute_error, mean_squared_error, root_mean_squared_error, max_error,
                             coverage_error,
                             mean_absolute_percentage_error, median_absolute_error,
                             mean_squared_log_error, root_mean_squared_log_error)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score,
                             d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
                             adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
                             jaccard_score, consensus_score, v_measure_score, brier_score_loss, d2_tweedie_score,
                             cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
                             average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
                             top_k_accuracy_score, calinski_harabasz_score, davies_bouldin_score)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc

# метрики ранжирования (рекомендательных систем)
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cityblock, cosine, euclidean, hamming, jaccard
from Levenshtein import distance as levenshtein

from sklearn.model_selection import train_test_split, GridSearchCV

import decorators
from tools import export2

SCALERS = (Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer)


def gini(y_true, y_predicted) -> float:
    """Критерий Джинни"""
    return 2 * roc_auc_score(y_true, y_predicted) - 1


def backward_importance_features(model, train: tuple, test: tuple,
                                 fit_parameters: dict | None = None) -> dict[str:tuple]:
    x_train, y_train = train
    x_test, y_test = test

    if fit_parameters is None: fit_parameters = dict()

    model.fit(x_train, y_train, **fit_parameters)
    history = {'': (model.score(x_train, y_train), model.score(x_test, y_test))}

    features = x_train.columns.to_list().copy()

    for feature in x_train.columns:
        features.remove(feature)
        model.fit(x_train[features], y_train, **fit_parameters)
        history[feature] = (model.score(x_train[features], y_train), model.score(x_test[features], y_test))
        features.append(feature)

    return dict(sorted(history.items(), key=lambda item: item[1][1]))


class Model:
    """Модель ML"""
    # линейные модели
    LINEAR_MODEL_CLASSIFIERS = [SGDClassifier, SGDOneClassSVM, RidgeClassifier, RidgeClassifierCV,
                                PassiveAggressiveClassifier]
    LINEAR_MODEL_REGRESSORS = [LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
                               OrthogonalMatchingPursuit,
                               BayesianRidge, ARDRegression, SGDRegressor, RANSACRegressor, GammaRegressor,
                               PoissonRegressor, HuberRegressor,
                               TweedieRegressor, LogisticRegression, QuantileRegressor, TheilSenRegressor]

    # деревья
    TREE_CLASSIFIERS = [DecisionTreeClassifier, ExtraTreeClassifier]
    TREE_REGRESSORS = [DecisionTreeRegressor, ExtraTreeRegressor]

    # ансамбли
    ENSEMBLE_CLASSIFIERS = [RandomForestClassifier, ExtraTreesClassifier,
                            BaggingClassifier,
                            GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
                            StackingClassifier,
                            VotingClassifier]
    ENSEMBLE_REGRESSORS = [RandomForestRegressor, ExtraTreesRegressor,
                           BaggingRegressor,
                           GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor,
                           StackingRegressor,
                           VotingRegressor]
    STACKINGS_CLASSIFIERS = [StackingClassifier]
    STACKINGS_REGRESSORS = [StackingRegressor]
    BAGGING_CLASSIFIERS = [BaggingClassifier]
    BAGGING_REGRESSORS = [BaggingRegressor]
    BOOSTING_CLASSIFIERS = [GradientBoostingClassifier]
    BOOSTING_REGRESSORS = [GradientBoostingRegressor]
    # соседи
    NEIGHBORS_CLASSIFIERS = [KNeighborsClassifier, RadiusNeighborsClassifier]
    NEIGHBORS_REGRESSORS = [KNeighborsRegressor, RadiusNeighborsRegressor]

    # стэкинги
    STACKINGS = STACKINGS_CLASSIFIERS + STACKINGS_REGRESSORS
    # бэггинги
    BAGGINGS = BAGGING_CLASSIFIERS + BAGGING_REGRESSORS
    # бустинги
    BOOSTINGS = BOOSTING_CLASSIFIERS + BOOSTING_REGRESSORS

    # классификаторы
    CLASSIFIERS = LINEAR_MODEL_CLASSIFIERS + TREE_CLASSIFIERS + ENSEMBLE_CLASSIFIERS
    # регрессоры
    REGRESSORS = LINEAR_MODEL_REGRESSORS + TREE_REGRESSORS + ENSEMBLE_REGRESSORS
    # кластеризаторы
    CLUSTERIZERS = NEIGHBORS_CLASSIFIERS + NEIGHBORS_REGRESSORS + [NearestNeighbors] + [DBSCAN]

    # все модели ML
    MODELS = set(CLASSIFIERS + REGRESSORS + CLUSTERIZERS)

    SCORES = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score,
              d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
              adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
              jaccard_score, consensus_score, v_measure_score, brier_score_loss, d2_tweedie_score,
              cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
              average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
              top_k_accuracy_score, calinski_harabasz_score, davies_bouldin_score]

    ERRORS = [mean_absolute_error, mean_squared_error, root_mean_squared_error, max_error,
              coverage_error,
              mean_absolute_percentage_error, median_absolute_error,
              mean_squared_log_error, root_mean_squared_log_error]

    def __init__(self, model=None):
        """Инициализация модели ML None/названием модели/объектом ML"""
        if model is None:
            self.__model = None
        elif type(model) in self.MODELS:
            self.__model = model
        else:
            raise AssertionError(
                f'type(model) in {[str.__name__] + [class_model.__name__ for class_model in self.MODELS]}')

    def __call__(self):
        return self.__model

    def __str__(self):
        return str(self.__model)

    @property
    def intercept_(self):
        try:
            return self.__model.intercept_
        except Exception as e:
            print(e)

    @property
    def coef_(self):
        try:
            return self.__model.coef_
        except Exception as e:
            print(e)

    @property
    def expression(self):
        try:
            return f'y = {self.intercept_} + {" + ".join([f"{num} * x{i + 1}" for i, num in enumerate(self.coef_)])}'
        except Exception as e:
            print(e)

    @property
    def feature_importances_(self):
        try:
            return self.__model.feature_importances_
        except Exception as e:
            print(e)

    def get_params(self, *args, **kwargs) -> dict:
        """
        Get parameters of the model.

        Parameters:
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

        Returns:
        Dict[str, Any]: Model parameters.
        """
        assert self.__model is not None, "Model is not defined."
        return self.__model.get_params(*args, **kwargs)

    def fit(self, x, y):
        """Обучение модели"""
        self.__model.fit(x, y)

    def predict(self, x):
        return self.__model.predict(x)

    def prediction(self, y_true, y_possible, suptitle='Prediction', bins=40, savefig=False):

        fg = plt.figure(figsize=(12, 8))
        plt.suptitle(suptitle, fontsize=14, fontweight='bold')
        gs = fg.add_gridspec(1, 2)

        fg.add_subplot(gs[0, 0])
        plt.grid(True)
        plt.hist(y_true - y_possible, bins=bins)

        fg.add_subplot(gs[0, 1])
        plt.grid(True)
        plt.scatter(y_true, y_possible, color='red')
        plt.plot(y_true, y_true, color='blue')

        if savefig: export2(plt, file_name=suptitle, file_extension='png')

    def report(self, y_true, y_pred) -> dict:
        """
        Generate a report of all metrics.

        Parameters:
        y_true (Any): True values.
        y_pred (Any): Predicted values.

        Returns:
        Dict[str, float]: Dictionary with metric names as keys and their values.
        """
        assert type(self).__name__ in ('Regressor', 'Classifier'), 'Only suport (Regressor, Classifier)'

        metrics = [func for func in type(self).__dict__ if not func.startswith("__") and func not in self.stop_methods]

        report = {}
        for metric in metrics:
            method = getattr(self, metric)
            try:
                result = method(y_true, y_pred)
                report[metric] = result
            except Exception as e:
                print(f"Error calculating {metric}: {e}")

        return report

    def plot_tree(self, **kwargs):
        try:
            plot_tree(self.__model, filled=kwargs.get('filled', True))
        except Exception as e:
            if e: print(e)

    @decorators.warns('ignore')
    def fit_all(self, x, y, exceptions=True):
        """Обучение всех моделей"""

        '''
        def fit_model(Model):
            return Model().fit(x, y)

        with mp.Pool(mp.cpu_count) as pool:
            results = pool.map(fit_model, self.MODELS)

        return results
        '''

        result = list()
        for class_model in tqdm(self.MODELS):
            try:
                model = class_model().fit(x, y)
                result.append(Model(model))
            except Exception as e:
                if exceptions: print(e)

        return result

    def errors(self, y_true, y_predict, exceptions=True) -> dict[str:float]:
        errors = dict()
        for error in self.ALL_ERRORS:
            try:
                errors[error.__name__] = error(y_true, y_predict)
            except Exception as e:
                if exceptions: print(e)
        return errors

    def scores(self, y_true, y_predict, exceptions=True) -> dict[str:float]:
        scores = dict()
        for score in self.ALL_SCORES:
            try:
                scores[score.__name__] = score(y_true, y_predict)
            except Exception as e:
                if exceptions: print(e)
        return scores

    # TODO
    def distances(self, x, y, exceptions=True) -> dict[str:float]:
        """"""
        pass

    def save(self, path: str) -> None:
        """Сохранение модели"""
        pickle.dump(self.__model, open(path, 'wb'))

    @classmethod
    def load(cls, path: str):
        """Загрузка модели"""
        return pickle.load(open(path, 'rb'))


class Classifier(Model):
    """Модель классификатора"""

    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)

    def confusion_matrix(self, y_true, y_predicted):
        """Матрица путаницы"""
        return confusion_matrix(y_true, y_predicted, labels=self.__model.classes_)

    def confusion_matrix_plot(self, y_true, y_predicted, title='confusion_matrix', **kwargs):
        """График матрицы путаницы"""
        cm = self.confusion_matrix(y_true, y_predicted)
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.__model.classes_)
        # plt.figure(figsize=kwargs.get('figsize', (12, 12)))
        # plt.title(title, fontsize=16, fontweight='bold')
        cmd.plot()
        plt.show()

    def precision_recall_curve(self, y_true, y_predicted, **kwargs):
        """График precision-recall"""
        precision, recall, threshold = precision_recall_curve(y_true, y_predicted)
        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.title(kwargs.get('title', 'precision recall curve'), fontsize=16, fontweight='bold')
        plt.plot(precision, recall, color='blue', label='PR')
        plt.legend(loc='lower left')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.xlabel('precision', fontsize=14)
        plt.ylabel('recall', fontsize=14)
        if kwargs.get('savefig', False):
            export2(plt, file_name=kwargs.get('title', 'precision recall curve'), file_extension='png')
        plt.show()

    def roc_curve(self, y_true, y_predicted, **kwargs):
        """График ROC"""
        fpr, tpr, threshold = roc_curve(y_true, y_predicted)
        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.title(kwargs.get('title', 'roc curve'), fontsize=16, fontweight='bold')
        plt.plot(fpr, tpr, color='blue', label=f'ROC-AUC = {auc(fpr, tpr)}')
        plt.plot([0, 1], [0, 1], color='red')
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.xlabel('fpr', fontsize=14)
        plt.ylabel('tpr', fontsize=14)
        if kwargs.get('savefig', False): export2(plt, file_name=kwargs.get('title', 'roc curve'), file_extension='png')
        plt.show()


class Regressor(Model):

    def __init__(self, *args, **kwargs):
        super(Regressor, self).__init__(*args, **kwargs)


class Clusterizer(Model):
    """Кластеризатор"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

    @property
    def labels_(self):
        return self.model.labels_

    @property
    def n_clusters(self):
        return self.model.n_clusters

    def elbow(self, x_train, max_k: int, change_n_clusters: bool = True, scale=False):
        """Метод локтя"""
        assert isinstance(max_k, int), f'Incorrect max_k param type. {type(max_k)} instead of {int}'
        assert self.model.__class__.__name__ in ('BisectingKMeans', 'KMeans', 'MiniBatchKMeans'), \
            f"This model doesn't support the elbow method. Valid models: {('BisectingKMeans', 'KMeans', 'MiniBatchKMeans')}"

        x_scaled = StandardScaler().fit_transform(x_train) if scale else x_train

        default_num_clusters = self.model.n_clusters

        wcss = []
        for k in range(1, max_k + 1, 1):
            self.model.n_clusters = k
            model = self.model.fit(x_scaled)
            wcss.append(model.inertia_)

        n_clust = self.__elbow_method_best_k(wcss)
        if change_n_clusters:
            self.model.n_clusters = n_clust
            self.model.fit(x_train)
            print(f"Your model's parameter 'n_clusters' was changed to optimal: {n_clust} and model was fitted on it.")
        else:
            self.model.n_clusters = default_num_clusters

        return wcss

    def elbow_plot(self, wcss) -> None:
        """Визуализация метода локтя"""
        assert isinstance(wcss, (list, tuple)), f'Incorrect wcss param type. {type(wcss)} instead of {list | tuple}'

        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.plot(range(1, len(wcss) + 1), wcss, marker='o', mfc='red')
        plt.title('Selecting the number of clusters using the elbow method')
        plt.xlabel('num clusters')
        plt.ylabel('WCSS (error)')
        plt.xticks(range(1, len(wcss) + 1))
        plt.show()

    def __elbow_method_best_k(self, wcss):
        """
        Determine the best number of clusters using the elbow method with a given threshold.

        Parameters:
        wcss (list or tuple): WCSS values for different numbers of clusters.

        Returns:
        int: Optimal number of clusters.
        """
        assert isinstance(wcss, (list, tuple)), f'Incorrect wcss parameter type. {type(wcss)} instead of {list | tuple}'
        assert len(wcss) >= 3, 'max_k len must be >= 3'

        # подробное описание работы алгоритма в файле про кластеризацию и метрики качества
        diff = np.diff(wcss)
        diff_r = diff[1:] / diff[:-1]
        k_opt = range(1, len(wcss))[np.argmin(diff_r) + 1]

        return k_opt

    def report(self, x_train, y_true, labels):
        """
        Calculate all clustering metrics for the given data and labels.

        Parameters:
        x_train (array-like): Training data.
        y_true (array-like): True labels.
        labels (array-like): Cluster labels.

        Returns:
        Dict[str, float]: Dictionary with metric names as keys and their values.
        """
        metrics = {
            'Silhouette Score': self.silhouette_score(x_train, labels),
            'Calinski-Harabasz Index': self.calinski_harabasz_score(x_train, labels),
            'Davies-Bouldin Index': self.davies_bouldin_score(x_train, labels),
            'Dunn Index': self.dunn_index(x_train, labels),
            'V-Measure': self.v_measure_score(y_true, labels),
            'Adjusted Rand Index': self.adjusted_rand_index(y_true, labels)
        }
        return metrics

    def __plot_dendrogram(self, model, **kwargs):
        """
        Generate the linkage matrix and plot the dendrogram.

        Parameters:
        model (object): Fitted clustering model.
        kwargs: Additional keyword arguments for the dendrogram plotting function.

        Returns:
        None
        """

        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)
        dendrogram(linkage_matrix, **kwargs)

    def dendrogram_plot(self, scale=False, **kwargs):
        """Визуализация дендограммы"""

        assert self.model.__class__.__name__ in ('AgglomerativeClustering'), f'Only support AgglomerativeClustering'
        assert hasattr(self.model, 'children_'), f'The model must be fitted'

        plt.figure(figsize=(10, 8))
        plt.title('Hierarchical Clustering Dendrogram')
        self.__plot_dendrogram(self.model, **kwargs)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()
        # или так
        x_scaled = StandardScaler().fit_transform(x_train) if scale else x_train

        clusters = self.model.fit_predict(x_scaled)
        plt.scater(x_scaled[:, 0], x_scaled[:, 1], cmap='autumn', c=clusters, s=60)

    def distances(self, scale=False, ):
        from scipy.spatial.distance import pdist
        from scipy.cluster import hierarchy
        x_scaled = StandardScaler().fit_transform(x_train) if scale else x_train
        distance_matrix = pdist(x_scaled)  # матрица попарных расстояний между точками
        z = hierarchy.linkage(distance_matrix, 'ward')  # тип расстояния Уорда
        plt.figure(figsize=(32, 16))
        dn = hierarchy.dendrogram(z)


# TODO: добавить в класс Model как статический метод
def classifier_or_regressor(model) -> str:
    model_type = list()
    try:
        model_type.append(type(model).__name__.lower())
    except:
        pass
    try:
        model_type.append(model.__name__.lower())
    except:
        pass

    if 'cla' in model_type: return 'cla'
    if 'reg' in model_type: return 'reg'
    return ''


class Stacking:

    # TODO: доделать
    def __init__(self, models: tuple | list):
        self.__stacking = StackingClassifier() if 'cla' in type(models).__name__.lower() else StackingRegressor()

    def __call__(self):
        return self.__stacking

    def fit(self, x, y):
        self.__stacking.fit(x, y)


class Bagging:

    def __init__(self, model, **kwargs):
        bagging_type = classifier_or_regressor(model)
        if bagging_type == 'cla':
            self.__bagging = BaggingClassifier(model, **kwargs)
        elif bagging_type == 'reg':
            self.__bagging = BaggingRegressor(model, **kwargs)
        else:
            raise 'type(model) is "classifier" or "regressor"'

    def __call__(self):
        return self.__bagging

    def fit(self, x, y):
        self.__bagging.fit(x, y)

    def predict(self, x):
        return self.__bagging.predict(x)


class Boosting(Classifier, Regressor):

    def __init__(self, model, *args, **kwargs):
        assert type(model) in self.BOOSTINGS
        if 'classifier' in model.__class__.__name__.lower():
            super(Boosting, self).__init__(model)
        elif 'regressor' in model.__class__.__name__.lower():
            super(Boosting, self).__init__(model)
        else:
            raise


class Ranking:
    """Ранжировщик рекомендательной системы"""

    def __init__(self, based='collab'):
        assert type(based) is str
        based = based.strip().lower()
        assert based in ('user', 'item', 'collab')
        self.based_on = based

    def fit(self, x, y, **kwargs):
        """"""
        assert len(x) == len(y)
        x, y = pd.DataFrame(x), pd.Series(y, name='target')  # категориальные целевые значения
        targets = y.unique().tolist()

        df = pd.concat([x, y], axis=1)
        group = df.groupby(y.name)  # группировка по целевым значениям
        # print(group.agg(['mean', 'count']).T)

        result = dict()
        for column in x.columns:
            result[column] = list()
            MEAN, MIN, MAX = group.count()[column].mean(), group.count()[column].min(), group.count()[column].max()
            delta = MAX - MIN
            # print(f'MEAN: {MEAN}, MIN: {MIN}, MAX: {MAX}')
            for target in targets:
                # print(group.mean().loc[target, column])
                # print(group.count().loc[target, column])
                # print()
                # взвешенные значения
                result[column].append((group.mean().loc[target, column] *
                                       (group.count().loc[target, column] - MEAN) / delta))
        df = pd.DataFrame(result)
        print(df)
        self.__model = NearestNeighbors(n_neighbors=kwargs.pop('n_neighbors', df.shape[0]),
                                        radius=kwargs.pop('radius', 1.0),
                                        algorithm="auto",
                                        leaf_size=30,
                                        metric=kwargs.pop('metric', "minkowski"),
                                        p=2,
                                        metric_params=None,
                                        n_jobs=None)
        self.__model.fit(df)  # подсчет расстояния
        return df

    def predict(self, x):
        """Предсказание"""
        result = self.__model.kneighbors(x, return_distance=True)
        return result[0]
        return x.iloc[result[1][0]]

    # TODO: доделать вывод всех метрик расстояний
    def report(self, y_true, y_pred):
        """Вывод всех метрик"""
        report = dict()
        for i, row in self.__model.iterrows():
            print(i, row.to_dict())
            print(cosine(y_true, y_pred))
        return report

    class RecommendSystem:
        def __init__(self, based_on=None) -> None:
            """
            Constructor to initialize the RecommendSystem class.

            Parameters:
            based_on (optional): An optional parameter that can be used to customize the initialization.
            """
            self.__based_on = based_on

        def fit(self, X, y, **kwargs) -> None:
            """
            Fit the recommendation model with provided data.

            Parameters:
            X (DataFrame): Feature matrix.
            y (Series): Target vector.
            **kwargs: Additional keyword arguments for configuring the NearestNeighbors model, such as:
                - n_neighbors (int): Number of neighbors to use. Default is the number of rows in the weighted average DataFrame.
                - radius (float): Range of parameter space to use by default for neighbors search. Default is 1.0.
                - metric (str): Metric to use for distance computation. Default is "minkowski".

            Returns:
            None: This method does not return any value. It fits the model with the provided data.
            """
            df = pd.concat([X, y], axis=1)
            weighted_avg = df.groupby(df.columns[-1]).apply(
                lambda g: g.iloc[:, :-1].multiply(len(g), axis=0).sum() / len(g))
            self.df = weighted_avg

            self.__model = NearestNeighbors(
                n_neighbors=kwargs.get('n_neighbors', self.df.shape[0]),
                radius=kwargs.get('radius', 1.0),
                algorithm="auto",
                leaf_size=30,
                metric=kwargs.get('metric', "minkowski"),
                p=2,
                metric_params=None,
                n_jobs=None
            )
            self.__model.fit(self.df)

        def predict(self, x, **kwargs):
            """
            Predict recommendations for the given input.

            Parameters:
            x (DataFrame or list): Input data for which recommendations are to be made. If a list is provided, it will be converted to a DataFrame.
            **kwargs: Additional keyword arguments for configuring the prediction, such as:
                - ascending (bool): Whether to sort the distances in ascending order. Default is False.

            Returns:
            list of DataFrames: Each DataFrame contains the recommendations and distances for the corresponding input.
            """
            if not isinstance(x, pd.DataFrame):
                x = pd.DataFrame([x])

            result = self.__model.kneighbors(x, return_distance=True)
            res_recomends = []
            for example, dist in zip(result[1], result[0]):
                temp_df = self.df.copy()
                temp_df['recommendation'] = example
                temp_df['distance'] = dist
                temp_df.sort_values('distance', inplace=True, ascending=kwargs.get('ascending', False))
                temp_df.reset_index(inplace=True, drop=True)
                res_recomends.append(temp_df)
            return res_recomends

        def levenshtein_distance_handmade(self, s1: str, s2: str) -> int:
            """
            Calculate the Levenshtein distance between two strings.

            Parameters:
            s1 (str): First string.
            s2 (str): Second string.

            Returns:
            int: The Levenshtein distance between the two strings.
            """
            len_s1 = len(s1)
            len_s2 = len(s2)

            dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

            for i in range(len_s1 + 1):
                dp[i][0] = i

            for j in range(len_s2 + 1):
                dp[0][j] = j

            for i in range(1, len_s1 + 1):
                for j in range(1, len_s2 + 1):
                    cost = 0 if s1[i - 1] == s2[j - 1] else 1
                    dp[i][j] = min(dp[i - 1][j] + 1,
                                   dp[i][j - 1] + 1,
                                   dp[i - 1][j - 1] + cost)

            return dp[len_s1][len_s2]

        def report(self, x, **kwargs):
            """
            Generate a report of recommendations using various distance metrics.

            Parameters:
            x (DataFrame or list): Input data for which the report is to be generated. If a list is provided, it will be converted to a DataFrame.
            **kwargs: Additional keyword arguments for configuring the report, such as:
                - n_neighbors (int): Number of neighbors to use. Default is the number of rows in the weighted average DataFrame.
                - radius (float): Range of parameter space to use by default for neighbors search. Default is 1.0.
                - sort_by (str): The distance metric to sort the recommendations by. Default is 'minkowski'.
                - ascending (bool): Whether to sort the distances in ascending order. Default is False.

            Returns:
            DataFrame: A DataFrame containing the recommendations and distances for the given input, sorted by the specified metric.
            """
            if not isinstance(x, pd.DataFrame):
                x = pd.DataFrame([x])

            recommendation_metrics = ['minkowski', 'cityblock', 'cosine', 'euclidean', 'haversine',
                                      'l1', 'l2', 'manhattan', 'nan_euclidean', levenshtein]
            temp_df = self.df.copy()
            for metric in recommendation_metrics:
                model = NearestNeighbors(
                    n_neighbors=kwargs.get('n_neighbors', self.df.shape[0]),
                    radius=kwargs.get('radius', 1.0),
                    algorithm="auto",
                    leaf_size=30,
                    metric=metric,
                    p=2,
                    metric_params=None,
                    n_jobs=None
                )
                model.fit(self.df)
                result = model.kneighbors(x, return_distance=True)
                name_metric = f'distance_{metric}' if metric != recommendation_metrics[-1] else f'distance_levenshtein'
                temp_df[name_metric] = result[0][0]
            temp_df.sort_values(f"distance_{kwargs.get('sort_by', 'minkowski')}", inplace=True,
                                ascending=kwargs.get('ascending', False))
            temp_df.reset_index(inplace=True, drop=True)
            return temp_df


if __name__ == '__main__':

    if 'Classifier' == '':
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        print(pd.concat([data.data, data.target], axis=1))

        classifier = Classifier()
        classifier.fit_all(data.data, data.target)

    if 'Regressor' == '':
        from sklearn.datasets import load_wine

        data = load_wine(as_frame=True)
        print(pd.concat([data.data, data.target], axis=1))

        regressor = Regressor()
        regressor.fit_all(data.data, data.target)

    if 'Clusterizer' == '':
        from sklearn.datasets import load_iris

        data = load_iris(as_frame=True)
        print(pd.concat([data.data, data.target], axis=1))

        clusterizer = Clusterizer()
        clusterizer.fit_all(data.data, data.target)

    if 'Stacking' == '':
        stacking = Stacking()

    if 'Bagging' == '':
        bagging = Bagging()

    if 'Boosting' == '':
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        print(pd.concat([data.data, data.target], axis=1))

        x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

        boosting = Boosting(GradientBoostingClassifier())
        boosting.fit(x_train, y_train)

        y_pred = boosting.predict(x_test)

        print(f'gini: {gini(y_test, y_pred)}')

    if 'GridSearchCV' == '':
        print(Fore.YELLOW + GridSearchCV.__name__ + Fore.RESET)

        from sklearn.datasets import load_breast_cancer
        from sklearn.datasets import load_wine
        from sklearn.datasets import load_iris

        datas = [load_breast_cancer(as_frame=True), load_wine(as_frame=True), load_iris(as_frame=True)]
        models = [GradientBoostingClassifier(), GradientBoostingRegressor(), DBSCAN()]
        param_grids = [
            {'n_estimators': list(range(25, 150 + 1, 25)),
             'max_depth': list(range(3, 15, 3)),
             'random_state': [0]},
            {'n_estimators': list(range(25, 150 + 1, 25)),
             'max_depth': list(range(3, 15, 3)),
             'random_state': [0]},
            {'n_estimators': list(range(25, 150 + 1, 25)),
             'max_depth': list(range(3, 15, 3)),
             'random_state': [0]}
        ]

        for i in range(3):
            x_train, x_test, y_train, y_test = train_test_split(datas[i].data, datas[i].target, test_size=0.25)

            gscv = GridSearchCV(models[i], param_grid=param_grids[i], n_jobs=-1)
            gscv.fit(x_train, y_train)

            print(f'best params: {gscv.best_params_}')
            best_model = gscv.best_estimator_
            print(f'best estimator: {best_model}')
            try:
                print(f'train score: {best_model.score(x_train, y_train)}')
                print(f'test score: {best_model.score(x_test, y_test)}')
            except:
                pass

    if 'Ranking' != '':
        '''
        from sklearn.datasets import load_wine

        data = load_wine(as_frame=True)
        df = pd.concat([data.data, data.target], axis=1)
        print(df)
        print(df.columns)
        '''

        df = pd.read_csv('ratings.csv')
        df.drop('timestamp', inplace=True, axis=1)

        if 'train_test_split':
            x_train, x_test, y_train, y_test = train_test_split(df.drop('movieId', axis=1), df['movieId'],
                                                                test_size=4, shuffle=True)
            print(x_train.shape, x_test.shape)

        rs = Ranking()
        x_scaled = rs.fit(x_train, y_train)
        print(x_scaled)
        y_pred = rs.predict(x_test)
        print(y_pred)
        exit()
        print(rs.report(y_test.iloc[0, :], y_pred.iloc[0, :]))

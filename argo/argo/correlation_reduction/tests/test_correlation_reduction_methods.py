import pandas as pd
import numpy as np
import pytest
from correlation_reduction.correlation_reduction_methods import AgglomerativeClusteringFeatureReduction
from correlation_reduction.similarity_functions import CosineSimilarity, JaccardSimilarity
from itertools import product


@pytest.fixture
def create_data():
    np.random.seed(0)
    X = pd.DataFrame({
        "A": np.random.randint(0, 2, 1000),
        "B": np.random.randint(0, 2, 1000),
        "C": np.random.randint(0, 2, 1000),
        "D": np.random.randint(0, 2, 1000),
        "E": np.random.randint(0, 2, 1000)
    })
    return X


@pytest.fixture
def similarity_df():
    similarity_df = pd.DataFrame({
        "index": ["A", "B", "C", "D", "E"],
        "A": [1.000000,	0.531925, 0.509824,	0.471540, 0.517857],
        "B": [0.531925, 1.000000, 0.508200,	0.521822, 0.526037],
        "C": [0.509824,	0.508200, 1.000000,	0.493806, 0.502041],
        "D": [0.471540,	0.521822, 0.493806,	1.000000, 0.483528],
        "E": [0.517857,	0.526037, 0.502041,	0.483528, 1.000000]
    })
    similarity_df.set_index('index', inplace=True)
    return similarity_df


@pytest.fixture
def columns_performance():
    columns_performance = pd.Series(
        [0.1, 0.2, 0.5, 0.4, 0.9],
        ["A", "B", "C", "D", "E"]
    )
    return columns_performance


@pytest.fixture
def clusters():
    clusters = pd.Series([0, 0, 0, 1, 0], ["A", "B", "C", "D", "E"])
    return clusters


@pytest.fixture
def cos_sim():
    cs = CosineSimilarity()
    return cs


@pytest.fixture
def jacc_sim():
    js = JaccardSimilarity()
    return js


@pytest.fixture
def agg_instantiated(columns_performance, cos_sim):
    columns_performance = columns_performance
    cs = cos_sim
    agg = AgglomerativeClusteringFeatureReduction(
        0.5, 'bottom_up', cs.fit, columns_performance)
    return agg


def test_fit(create_data, agg_instantiated, cos_sim, jacc_sim, columns_performance):
    expected_results = [
        ['E', 'D'],
        ['E', 'D'],
        ['E'],
        ['E'],
        ['D', 'A', 'B', 'C', 'D', 'E'],
        ['D', 'A', 'B', 'C', 'D', 'E'],
        ['A', 'B', 'C', 'D', 'E'],
        ['A', 'B', 'C', 'D', 'E']
    ]
    X = create_data
    cs = cos_sim
    js = jacc_sim
    columns_performance = columns_performance
    combinations = list(
        product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit]))
    for i, (threshold, strategy, similarity_function) in enumerate(combinations):
        fr = AgglomerativeClusteringFeatureReduction(
            threshold, strategy, similarity_function, columns_performance)
        fr.fit(X)
        assert fr.columns_to_keep == expected_results[i]


def test_transform(create_data, agg_instantiated, cos_sim, jacc_sim, columns_performance):
    expected_results = [2, 2, 1, 1, 6, 6, 5, 5]
    X = create_data
    cs = cos_sim
    js = jacc_sim
    columns_performance = columns_performance
    combinations = list(
        product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit]))
    for i, (threshold, strategy, similarity_function) in enumerate(combinations):
        fr = AgglomerativeClusteringFeatureReduction(
            threshold, strategy, similarity_function, columns_performance)
        fr.fit(X)
        X_reduced = fr.transform(X)
        assert X_reduced.shape[1] == expected_results[i]


def test_fit_transform(create_data, agg_instantiated, cos_sim, jacc_sim, columns_performance):
    expected_cols_to_keep = [
        ['E', 'D'],
        ['E', 'D'],
        ['E'],
        ['E'],
        ['D', 'A', 'B', 'C', 'D', 'E'],
        ['D', 'A', 'B', 'C', 'D', 'E'],
        ['A', 'B', 'C', 'D', 'E'],
        ['A', 'B', 'C', 'D', 'E']
    ]
    expected_col_num_results = [2, 2, 1, 1, 6, 6, 5, 5]
    X = create_data
    cs = cos_sim
    js = jacc_sim
    columns_performance = columns_performance
    combinations = list(
        product([0.25, 0.75], ['top_down', 'bottom_up'], [cs.fit, js.fit]))
    for i, (threshold, strategy, similarity_function) in enumerate(combinations):
        fr = AgglomerativeClusteringFeatureReduction(
            threshold, strategy, similarity_function, columns_performance)
        X_reduced = fr.fit_transform(X)
        assert fr.columns_to_keep == expected_cols_to_keep[i]
        assert X_reduced.shape[1] == expected_col_num_results[i]


def test_bottom_up(agg_instantiated, clusters, similarity_df):
    agg = agg_instantiated
    clusters = clusters
    similarity_df = similarity_df
    cols_to_drop = agg._bottom_up(clusters, int(
        similarity_df.shape[0]/2), similarity_df)
    assert cols_to_drop == ['A', 'B', 'C']
    assert agg.columns_to_keep == []


def test_top_down(agg_instantiated, clusters, similarity_df):
    agg = agg_instantiated
    clusters = clusters
    similarity_df = similarity_df
    cols_to_drop = agg._top_down(clusters, 2, similarity_df)
    assert cols_to_drop == ['A', 'B', 'C', 'E']
    assert agg.columns_to_keep == ['E', 'D']


def test_set_n_clusters(agg_instantiated, similarity_df):
    agg = agg_instantiated
    similarity_df = similarity_df
    for strategy in ['top_down', 'bottom_up']:
        agg.strategy = strategy
        n_clusters = agg._set_n_clusters(similarity_df)
        assert n_clusters == 2


def test_calculate_cluster_median(agg_instantiated, similarity_df):
    agg = agg_instantiated
    similarity_df = similarity_df
    cluster_median = agg._calculate_cluster_median(similarity_df)
    assert round(cluster_median, 6) == 0.509012


def test_get_top_performer(agg_instantiated, columns_performance):
    agg = agg_instantiated
    columns_performance = columns_performance
    top_performer = agg._get_top_performer(['A', 'B'], columns_performance)
    assert top_performer == 'B'


def test_agglomerative_clustering(similarity_df, agg_instantiated):
    similarity_df = similarity_df
    agg = agg_instantiated
    clusters = agg._agglomerative_clustering(similarity_df, 2)
    assert all(clusters.values == np.array([0, 0, 0, 1, 0]))

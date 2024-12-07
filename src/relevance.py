import numpy as np
import pandas as pd
from typing import Union

from src.ranker import Ranker


def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_results: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant.
        cut_off: The search result rank to stop calculating MAP. The default cut-off is 10;
            calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    cut_off = min(cut_off, len(search_result_relevances))

    n_rel_seen = 0
    precision_at_recall_points = []
    for i in range(cut_off):
        if search_result_relevances[i]:
            n_rel_seen += 1
            precision_at_recall_points.append(n_rel_seen / (i + 1))

    if n_rel_seen == 0:  # no relevant documents within the cut-off
        return 0

    n_rel_total = sum(search_result_relevances)
    return sum(precision_at_recall_points) / n_rel_total


def ndcg_score(search_result_relevances: list[float],
               ideal_relevance_score_ordering: list[float],
               cut_off: int = 10) -> float:
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned. These are the human-derived document relevance scores,
            *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by
            relevance score in descending order. Use this list to calculate IDCG (Ideal DCG).
        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    if not search_result_relevances or not ideal_relevance_score_ordering:
        return 0

    cut_off = min(cut_off, len(search_result_relevances))

    dcg = search_result_relevances[0]
    idcg = ideal_relevance_score_ordering[0]

    for i in range(1, cut_off):
        disc = np.log2(i + 1)
        dcg += search_result_relevances[i] / disc
        if i < len(ideal_relevance_score_ordering):
            idcg += ideal_relevance_score_ordering[i] / disc

    return dcg / idcg


def run_relevance_tests(relevance_data_filename: str,
                        ranker: Ranker,
                        encoding: str = "utf-8",
                        cut_off: int = 10) -> dict[str, Union[float, list[float]]]:
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.

    Args:
        relevance_data_filename [str]: The filename containing the relevance data to be loaded.
        ranker: A ranker configured with a particular scoring function to search through the document collection.
                This is probably either a Ranker or a L2RRanker object, but something that has a query() method.
        encoding: The encoding of the relevance data file. The default is "utf-8".
        cut_off: The search result rank to stop calculating MAP@K and NDCG@K. The default cut-off is 10.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # Load the relevance dataset
    relevance_df = pd.read_csv(relevance_data_filename, encoding=encoding)

    # Run each of the dataset's queries through the ranking function
    # For each query's result, calculate the MAP and NDCG for every single query and average them out
    # NOTE: MAP requires using binary judgments of relevant (1) or not (0).
    #   Consider relevance scores of (1,2,3) as not-relevant and (4,5) as relevant.
    # NOTE: NDCG can use any scoring range, so no conversion is needed.

    avps: list[float] = []
    ndcgs: list[float] = []

    for query, qdf in relevance_df.groupby("query"):
        # Get all relevance judgements
        relevances = dict(zip(qdf.docid, qdf.rel))

        # Retrieved ranking
        ranker_results = ranker.query(query)
        ranker_relevances_bin = []  # for MAP
        ranker_relevances = []  # for NDCG
        for docid, _  in ranker_results:
            doc_rel = relevances.get(docid, 0)
            ranker_relevances_bin.append(int(doc_rel > 3))
            ranker_relevances.append(doc_rel)

        # Calculate Average Precision
        avp = map_score(ranker_relevances_bin, cut_off=cut_off)

        # Calculate NDCG
        ideal_relevances = sorted(relevances.values(), reverse=True)
        ndcg = ndcg_score(ranker_relevances, ideal_relevances, cut_off=cut_off)

        avps.append(avp)
        ndcgs.append(ndcg)

    # Compute the average MAP and NDCG across all queries
    return {'map': np.mean(avps), 'ndcg': np.mean(ndcgs), 'map_list': avps, 'ndcg_list': ndcgs}


if __name__ == '__main__':
    pass

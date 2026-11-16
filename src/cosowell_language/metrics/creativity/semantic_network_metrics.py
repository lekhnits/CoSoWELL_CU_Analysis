"""
Family 6 - semantic network: graph of consecutive content-word lemmas
(speech-graph approach, Mota et al., 2012).

    density         share of all possible word pairs that are linked
    modularity      how strongly the graph falls apart into separate groups of words
    clustering      how often two neighbours of a word are also neighbours of each other
    lcc share       share of the words in the largest connected part of the graph
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .picture_corpus import PictureCorpus

COLUMNS = ["network_density", "network_modularity", "network_avg_path_length", "network_clustering",
           "network_lcc_share"]


def word_graph_metrics(content: list) -> dict:
    """Nodes are the content-word lemmas; an edge joins two lemmas that follow each other in the text."""
    import networkx as nx

    lemmas = [token.lemma_.lower() for token in content]
    graph = nx.Graph()
    graph.add_nodes_from(lemmas)
    graph.add_edges_from((lemma, next_lemma) for lemma, next_lemma in zip(lemmas, lemmas[1:]) if lemma != next_lemma)
    if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
        return dict.fromkeys(COLUMNS, np.nan)
    largest_part = graph.subgraph(max(nx.connected_components(graph), key=len))
    communities = nx.community.greedy_modularity_communities(graph)
    return {
        "network_density": nx.density(graph),
        "network_modularity": nx.community.modularity(graph, communities),
        # Path length is only defined within a connected part, so it uses the largest one.
        "network_avg_path_length": nx.average_shortest_path_length(largest_part) if len(largest_part) > 1 else np.nan,
        "network_clustering": nx.average_clustering(graph),
        "network_lcc_share": len(largest_part) / graph.number_of_nodes(),
    }


def semantic_network_metrics(corpus: PictureCorpus) -> pd.DataFrame:
    from tqdm.auto import tqdm

    rows = [word_graph_metrics(content) for content in tqdm(corpus.content_words, desc="graphs")]
    return pd.DataFrame(rows, columns=COLUMNS, index=corpus.narratives.index)

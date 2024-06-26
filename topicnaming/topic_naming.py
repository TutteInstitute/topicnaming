import numpy as np
import pandas as pd
import datasets
import fast_hdbscan
from fast_hdbscan import cluster_trees, hdbscan, numba_kdtree, boruvka

import numba

import sklearn.metrics
import vectorizers
import vectorizers.transformers
import sklearn.feature_extraction
import scipy.sparse
import warnings
from random import sample

import sentence_transformers

from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
from dataclasses import dataclass

from tqdm.auto import tqdm
import string


@numba.njit(fastmath=True)
def layer_from_clustering(
    point_vectors,
    point_locations,
    cluster_label_vector,
    cluster_membership_vector,
    base_clusters,
    membership_strength_threshold=0.2,
):
    n_clusters = len(set(cluster_label_vector)) - 1

    average_vectors = np.zeros((n_clusters, point_vectors.shape[1]), dtype=np.float32)
    average_locations = np.zeros(
        (n_clusters, point_locations.shape[1]), dtype=np.float32
    )
    total_weights = np.zeros(n_clusters, dtype=np.float32)
    pointsets = [set([-1 for i in range(0)]) for i in range(n_clusters)]
    metaclusters = [set([-1 for i in range(0)]) for i in range(n_clusters)]

    for i in range(cluster_label_vector.shape[0]):
        cluster_num = cluster_label_vector[i]
        if cluster_num >= 0:
            average_vectors[cluster_num] += (
                cluster_membership_vector[i] * point_vectors[i]
            )
            average_locations[cluster_num] += (
                cluster_membership_vector[i] * point_locations[i]
            )
            total_weights[cluster_num] += cluster_membership_vector[i]

            if cluster_membership_vector[i] > membership_strength_threshold:
                pointsets[cluster_num].add(i)
                sub_cluster = base_clusters[i]
                if sub_cluster != -1:
                    metaclusters[cluster_num].add(sub_cluster)

    for c in range(n_clusters):
        average_vectors[c] /= total_weights[c]
        average_locations[c] /= total_weights[c]

    return average_vectors, average_locations, pointsets, metaclusters


def build_cluster_layers(
    point_vectors,
    point_locations,
    *,
    min_clusters=2,
    min_samples=5,
    base_min_cluster_size=10,
    membership_strength_threshold=0.2,
    next_cluster_size_quantile=0.8,
    verbose=False,
):
    vector_layers = []
    location_layers = []
    pointset_layers = []
    metacluster_layers = []

    min_cluster_size = base_min_cluster_size

    sklearn_tree = hdbscan.KDTree(point_locations)
    numba_tree = numba_kdtree.kdtree_to_numba(sklearn_tree)
    edges = boruvka.parallel_boruvka(
        numba_tree, min_samples=min_cluster_size if min_samples is None else min_samples
    )
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = cluster_trees.mst_to_linkage_tree(sorted_mst)
    new_tree = cluster_trees.condense_tree(uncondensed_tree, base_min_cluster_size)
    leaves = cluster_trees.extract_leaves(new_tree)
    clusters = cluster_trees.get_cluster_label_vector(new_tree, leaves, 0.0)
    point_probs = cluster_trees.get_point_membership_strength_vector(
        new_tree, leaves, clusters
    )

    cluster_ids = np.unique(clusters[clusters >= 0])
    base_clusters = clusters.copy()
    n_clusters_in_layer = cluster_ids.shape[0]

    base_layer = True

    while n_clusters_in_layer >= min_clusters:

        layer_vectors, layer_locations, layer_pointsets, layer_metaclusters = (
            layer_from_clustering(
                point_vectors,
                point_locations,
                clusters,
                point_probs,
                base_clusters,
                membership_strength_threshold,
            )
        )

        if not base_layer:
            layer_metacluster_selection = np.asarray(
                [len(x) > 1 for x in layer_metaclusters]
            )
            layer_metaclusters = [
                list(x)
                for x, select in zip(layer_metaclusters, layer_metacluster_selection)
                if select
            ]
            layer_pointsets = [
                list(x)
                for x, select in zip(layer_pointsets, layer_metacluster_selection)
                if select
            ]
            layer_vectors = layer_vectors[layer_metacluster_selection]
            layer_locations = layer_locations[layer_metacluster_selection]

        vector_layers.append(layer_vectors)
        location_layers.append(layer_locations)
        pointset_layers.append(layer_pointsets)
        metacluster_layers.append(layer_metaclusters)

        last_min_cluster_size = min_cluster_size
        min_cluster_size = int(
            np.quantile([len(x) for x in layer_pointsets], next_cluster_size_quantile)
        )
        if verbose:
            print(
                f"cluster={len(layer_vectors)}, last_min_cluster_size={last_min_cluster_size}, min_cluster_size={min_cluster_size}"
            )

        new_tree = cluster_trees.condense_tree(uncondensed_tree, min_cluster_size)
        leaves = cluster_trees.extract_leaves(new_tree)
        clusters = cluster_trees.get_cluster_label_vector(new_tree, leaves, 0.0)
        point_probs = cluster_trees.get_point_membership_strength_vector(
            new_tree, leaves, clusters
        )

        cluster_ids = np.unique(clusters[clusters >= 0])
        n_clusters_in_layer = np.max(clusters) + 1
        base_layer = False

    pointset_layers = [
        [list(pointset) for pointset in layer] for layer in pointset_layers
    ]
    return vector_layers, location_layers, pointset_layers, metacluster_layers


def diversify(query_vector, candidate_neighbor_vectors, alpha=1.0, max_candidates=16):
    distance_to_query = np.squeeze(
        sklearn.metrics.pairwise_distances(
            [query_vector], candidate_neighbor_vectors, metric="cosine"
        )
    )

    retained_neighbor_indices = [0]
    for i, vector in enumerate(candidate_neighbor_vectors[1:], 1):
        retained_neighbor_distances = sklearn.metrics.pairwise_distances(
            [vector],
            candidate_neighbor_vectors[retained_neighbor_indices],
            metric="cosine",
        )[0]
        for j in range(retained_neighbor_distances.shape[0]):
            if alpha * distance_to_query[i] > retained_neighbor_distances[j]:
                break
        else:
            retained_neighbor_indices.append(i)
            if len(retained_neighbor_indices) >= max_candidates:
                return retained_neighbor_indices

    return retained_neighbor_indices


def topical_sentences_for_cluster(
    docs, vector_array, pointset, centroid_vector, n_sentence_examples=16
):
    sentences = docs.values[pointset]

    sent_vectors = vector_array[pointset]
    candidate_neighbor_indices = np.argsort(
        np.squeeze(
            sklearn.metrics.pairwise_distances(
                [centroid_vector], sent_vectors, metric="cosine"
            )
        )
    )
    candidate_neighbors = sent_vectors[candidate_neighbor_indices]
    topical_sentence_indices = candidate_neighbor_indices[
        diversify(centroid_vector, candidate_neighbors)[:n_sentence_examples]
    ]
    topical_sentences = [sentences[i] for i in topical_sentence_indices]
    return topical_sentences


def distinctive_sentences_for_cluster(
    cluster_num,
    docs,
    vector_array,
    pointset_layer,
    cluster_neighbors,
    n_sentence_examples=16,
):
    pointset = pointset_layer[cluster_num]
    sentences = docs.values[pointset]

    local_vectors = vector_array[
        sum([pointset_layer[x] for x in cluster_neighbors], [])
    ]
    vectors_for_svd = normalize(local_vectors - local_vectors.mean(axis=0))
    U, S, Vh = randomized_svd(
        vectors_for_svd, min(int(np.sqrt(vectors_for_svd.shape[0])), 64)
    )
    transformed_docs = local_vectors @ Vh.T
    transformed_docs = np.maximum(transformed_docs, 0)
    class_labels = np.repeat(
        np.arange(len(cluster_neighbors)),
        [len(pointset_layer[x]) for x in cluster_neighbors],
    )
    iwt = vectorizers.transformers.InformationWeightTransformer().fit(
        transformed_docs, class_labels
    )
    sentence_weights = np.sum(
        transformed_docs[: len(pointset)] * iwt.information_weights_, axis=1
    )
    distinctive_sentence_indices = np.argsort(sentence_weights)[
        : n_sentence_examples * 3
    ]
    distinctive_sentence_vectors = vector_array[distinctive_sentence_indices]
    diversified_candidates = diversify(
        vector_array[pointset_layer[cluster_num]].mean(axis=0),
        distinctive_sentence_vectors,
    )
    distinctive_sentence_indices = distinctive_sentence_indices[
        diversified_candidates[:n_sentence_examples]
    ]
    distinctive_sentences = [sentences[i] for i in distinctive_sentence_indices]
    return distinctive_sentences


def longest_keyphrases(candidate_keyphrases):
    result = []
    for i, phrase in enumerate(candidate_keyphrases):
        for other in candidate_keyphrases:
            if f" {phrase}" in other or f"{phrase} " in other:
                phrase = other

        if phrase not in result:
            candidate_keyphrases[i] = phrase
            result.append(phrase)

    return result


def create_final_remedy_prompt(
    original_topic_names,
    docs,
    vector_array,
    pointset,
    centroid_vector,
    doc_type,
    corpus_type,
    llm_instruction="A better and more specific name that still captures the topic of these article titles is:",
):
    sentences = topical_sentences_for_cluster(
        docs, vector_array, pointset, centroid_vector, n_sentence_examples=64
    )
    prompt_text = (
        f"A set of {doc_type} from {corpus_type} was described as having a topic of one of "
        + ", ".join(original_topic_names)
        + ".\n"
    )
    prompt_text += "These topic names were not specific enough and were shared with other different but similar groups of titles.\n"
    prompt_text += "A sampling of titles from this specific set of titles includes:\n"
    for sentence in np.random.choice(
        sentences, size=min(len(sentences), 64), replace=False
    ):
        prompt_text += f"- {sentence}\n"

    prompt_text += f"\n\nThe current name for this topic of these paragraphs is: {original_topic_names[-1]}\n"
    prompt_text += "\n" + llm_instruction
    return prompt_text


@dataclass
class ClusterLayers:
    """Class for keeping track of cluster layer information"""

    vector_layers: list[list[list]]
    location_layers: list[list[list]]
    pointset_layers: list[list[list]]
    metacluster_layers: list[list[list]]
    layer_cluster_neighbours: list[list[list]]


class TopicNaming:
    """
    documents: list of strings
        A list of objects to topic model.  Our current LLM topic naming functions currently presume these to be strings.
    document_vectors: numpy array
        A numpy array of shape number_of_objects by features.  These are vectors which encode the semantic similarity of our
        documents being topic modeled.
    document_map: numpy array
        A numpy array of shape number_of_objects by 2 (or 3).  These are two dimensional vectors often corresponding
        to a 2 dimensional umap of the document_vectors.
    cluster_layers: list of lists (optional, default None):
        A list with one element for each layer in your hierarchical clustering.
        Each layer is a list
    representative_sentences: dict (optional, default None):
        A dictionary from one of a set of ways to represent a document cluster to a the cluster representation.
    trim_percentile: int (between 0 and 100)
        Trim any document with a token length longer than the 99th percentile. This prevents very long outlier documents from swamping our prompts.
        The trim length will be the maximum of this value and trim_length.  Set to 100 if you don't want any trimming.
    trim_length: int
        Maximum number of tokens to keep from each document. This prevents very long outlier documents from swamping our prompts.
        The trim length will be the maximum of this value and trim_length. Set to None if you don't want any trimming.
    """

    def __init__(
        self,
        documents,
        document_vectors,
        document_map,
        llm,
        embedding_model=None,  # The embedding model that the document_vectors was constructed with.
        cluster_layers=None,  # ClusterLayer dataclass
        representation_techniques=["topical", "distinctive", "contrastive"],
        document_type="titles",
        corpus_description="academic articles",
        verbose=True,
        trim_percentile=99,
        trim_length=100,
        keyphrase_min_occurrences=25,
        keyphrase_ngram_range=(1, 4),
        n_sentence_examples_per_cluster=16,
        n_keyphrases_per_cluster=16,
        max_subtopics_per_cluster=32,
        max_neighbors_per_cluster=6,
    ):
        self.documents = documents
        self.document_vectors = document_vectors
        self.document_map = document_map
        if (cluster_layers is not None) and (
            type(cluster_layers).__name__ != "ClusterLayers"
        ):
            raise ValueError(
                f"cluster_layers must be of type ClusterLayers class not {type(cluster_layers).__name__}"
            )
        if cluster_layers:
            self.cluster_layers_ = cluster_layers
        self.representation_techniques = representation_techniques
        self.embedding_model = embedding_model
        # Check that this is either None or has an embed function.
        self.document_type = document_type
        self.corpus_description = corpus_description
        self.llm = llm
        self.verbose = verbose
        self.trim_percentile = trim_percentile
        self.trim_length = trim_length
        self.keyphrase_min_occurrences = keyphrase_min_occurrences
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.n_sentence_examples_per_cluster = n_sentence_examples_per_cluster
        self.n_keyphrases_per_cluster = n_keyphrases_per_cluster
        self.max_subtopics_per_cluster = max_subtopics_per_cluster
        self.max_neighbors_per_cluster = max_neighbors_per_cluster
        # Determine trim length used
        self.token_distribution = [len(llm.tokenize(text)) for text in documents]
        self.token_trim_length = int(
            np.percentile(self.token_distribution, trim_percentile)
        )
        if trim_length:
            self.token_trim_length = np.max([self.token_trim_length, trim_length])
        if trim_length > self.llm.n_ctx():
            warnings.warn(
                f"trim_length of {self.token_trim_length} > max context window {self.llm.n_ctx()} setting it to half of the maximum context window."
            )
            self.token_trim_length = self.llm.n_ctx() // 2

    def _trim_text(self, text):
        tokenized = self.llm.tokenize(text)
        return self.llm.detokenize(tokenized[: self.token_trim_length])

    def fit_clusters(self, base_min_cluster_size=100, min_clusters=6):
        """
        Constructs a layered hierarchical clustering well suited for layered topic modeling.
        TODO: Add a check to ensure that there were any cluster generated at the specified base_min_cluster_size.
        """
        if self.verbose:
            print(f"constructing cluster layers")
        self.base_min_cluster_size_ = base_min_cluster_size
        self.min_clusters_ = min_clusters

        vector_layers, location_layers, pointset_layers, metacluster_layers = (
            build_cluster_layers(
                self.document_vectors,
                self.document_map,
                base_min_cluster_size=base_min_cluster_size,
                min_clusters=min_clusters,
                verbose=self.verbose,
            )
        )

        layer_cluster_neighbours = [
            np.argsort(
                sklearn.metrics.pairwise_distances(layer, metric="cosine"), axis=1
            )[:, : self.max_neighbors_per_cluster]
            for layer in vector_layers
        ]
        self.cluster_layers_ = ClusterLayers(
            vector_layers,
            location_layers,
            pointset_layers,
            metacluster_layers,
            layer_cluster_neighbours,
        )

    def get_topical_layers(self, n_sentence_examples=16):
        """
        Fits a set of topical documents to describe a cluster.
        If the cluster_layers_ have not yet been generated or is None it will generate them as necessary.
        """
        # Call it yourself or get the default parameter choice.
        # Maybe throw a warning.
        if getattr(self, "cluster_layers_", None) is None:
            self.fit_clusters()

        topical_sentences_per_cluster = [
            [
                topical_sentences_for_cluster(
                    self.documents,
                    self.document_vectors,
                    pointset,
                    cluster_vector,
                    n_sentence_examples=n_sentence_examples,
                )
                for pointset, cluster_vector in tqdm(
                    zip(
                        self.cluster_layers_.pointset_layers[i],
                        self.cluster_layers_.vector_layers[i],
                    ),
                    desc=f"Topical sentences for layer {i}",
                    total=len(self.cluster_layers_.pointset_layers[i]),
                    disable=(not self.verbose),
                )
            ]
            for i in range(len(self.cluster_layers_.pointset_layers))
        ]
        return topical_sentences_per_cluster

    def get_distinctive_layers(self, n_sentence_examples=16):
        """
        Fits a set of distincts documents to describe a cluster.
        If the cluster_layers_ have not yet been generated or is None it will generate them as necessary.
        """
        # Call it yourself or get the default parameter choice.
        # Maybe throw a warning.

        if getattr(self, "cluster_layers_", None) is None:
            self.fit_clusters()

        distinctive_sentences_per_cluster = [
            [
                distinctive_sentences_for_cluster(
                    topic_num,
                    self.documents,
                    self.document_vectors,
                    self.cluster_layers_.pointset_layers[i],
                    self.cluster_layers_.layer_cluster_neighbours[i][topic_num],
                    n_sentence_examples=n_sentence_examples,
                )
                for topic_num in tqdm(
                    range(len(self.cluster_layers_.pointset_layers[i])),
                    desc=f"Distinctive sentences for layer {i}",
                    disable=(not self.verbose),
                )
            ]
            for i in range(len(self.cluster_layers_.pointset_layers))
        ]
        return distinctive_sentences_per_cluster

    def _contrastive_keywords_for_layer(
        self,
        layer_num,
        full_count_matrix,
        inverse_vocab,
        vocab_vectors,
        n_keywords=16,
        prior_strength=0.1,
        weight_power=2.0,
    ):
        pointset_layer = self.cluster_layers_.pointset_layers[layer_num]
        count_matrix = full_count_matrix[sum(pointset_layer, []), :]
        column_mask = np.squeeze(np.asarray(count_matrix.sum(axis=0))) > 0.0
        count_matrix = count_matrix[:, column_mask]
        column_map = np.arange(full_count_matrix.shape[1])[column_mask]
        row_mask = np.squeeze(np.asarray(count_matrix.sum(axis=1))) > 0.0
        count_matrix = count_matrix[row_mask, :]
        bad_rows = set(np.where(~row_mask)[0])

        class_labels = np.repeat(
            np.arange(len(pointset_layer)), [len(x) for x in pointset_layer]
        )[row_mask]
        iwt = vectorizers.transformers.InformationWeightTransformer(
            prior_strength=prior_strength, weight_power=weight_power
        ).fit(count_matrix, class_labels)
        count_matrix.data = np.log(count_matrix.data + 1)
        count_matrix.eliminate_zeros()

        weighted_matrix = iwt.transform(count_matrix)

        contrastive_keyword_layer = []

        from_row = 0

        for i in range(len(pointset_layer)):
            if i in bad_rows:
                contrastive_keyword_layer.append(["no keywords were found"])
            else:
                to_row = from_row + len(pointset_layer[i])
                contrastive_scores = np.squeeze(
                    np.asarray(weighted_matrix[from_row:to_row].sum(axis=0))
                )
                contrastive_keyword_indices = np.argsort(contrastive_scores)[
                    -4 * n_keywords :
                ]
                contrastive_keywords = [
                    inverse_vocab[column_map[j]]
                    for j in reversed(contrastive_keyword_indices)
                ]
                contrastive_keywords = longest_keyphrases(contrastive_keywords)

                centroid_vector = np.mean(
                    self.document_vectors[pointset_layer[i]], axis=0
                )
                keyword_vectors = np.asarray(
                    [vocab_vectors[word] for word in contrastive_keywords]
                )
                chosen_indices = diversify(
                    centroid_vector, keyword_vectors, alpha=0.66
                )[:n_keywords]
                contrastive_keywords = [contrastive_keywords[j] for j in chosen_indices]

                contrastive_keyword_layer.append(contrastive_keywords)
                from_row = to_row

        return contrastive_keyword_layer

    def get_contrastive_keyword_layers(self, n_keyphrases_per_cluster=16):
        """
        Fits a set of contrastive keywords to describe a cluster.
        If the cluster_layers_ have not yet been generated or is None it will generate them as necessary.
        """
        # TODO: count_vectorizer: CountVectorizer might be passed in at some point but for now is hard coded.
        # Call it yourself or get the default parameter choice.
        # Maybe throw a warning.

        if getattr(self, "cluster_layers_", None) is None:
            self.fit_clusters()
        # Check if embedding_model is set and has an encode function

        cv = sklearn.feature_extraction.text.CountVectorizer(
            lowercase=True,
            min_df=self.keyphrase_min_occurrences,
            token_pattern="(?u)\\b\\w[-'\\w]+\\b",
            ngram_range=self.keyphrase_ngram_range,
        )
        full_count_matrix = cv.fit_transform(self.documents)
        acceptable_vocab = [
            v
            for v in cv.vocabulary_
            if v.split()[0] not in sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
            and v.split()[-1] not in sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
        ]
        acceptable_indices = [cv.vocabulary_[v] for v in acceptable_vocab]
        full_count_matrix = full_count_matrix[:, acceptable_indices]
        inverse_vocab = {i: w for i, w in enumerate(acceptable_vocab)}
        vocab = acceptable_vocab

        if self.verbose:
            print(
                f"Created a potential keyphrase vocabulary of {len(vocab)} potential keyphrases"
            )

        vocab_vectors = dict(
            zip(
                vocab,
                self.embedding_model.encode(vocab, show_progress_bar=self.verbose),
            )
        )

        contrastive_keyword_layers = [
            self._contrastive_keywords_for_layer(
                layer_num,
                full_count_matrix,
                inverse_vocab,
                vocab_vectors,
                n_keywords=n_keyphrases_per_cluster,
            )
            for layer_num in tqdm(
                range(len(self.cluster_layers_.pointset_layers)),
                desc=f"Finding contrastive keywords",
                disable=(not self.verbose),
            )
        ]
        return contrastive_keyword_layers

    # Might use a dict for handling options
    def fit_representation(self):
        """
        Samples topical_layers, distincive_layers and contrastive_keyword layers depending on which methods have been included in the representation_techniques.
        If the cluster_layers_ have not yet been generated or is None it will generate them as necessary.
        """

        if getattr(self, "cluster_layers_", None) is None:
            self.fit_clusters()
        if self.verbose:
            print(f"sampling documents per cluster")
        self.representation_ = dict()
        for rep in self.representation_techniques:
            if rep == "topical":
                self.representation_[rep] = self.get_topical_layers(
                    n_sentence_examples=self.n_sentence_examples_per_cluster
                )
            elif rep == "distinctive":
                self.representation_[rep] = self.get_distinctive_layers(
                    n_sentence_examples=self.n_sentence_examples_per_cluster
                )
            elif rep == "contrastive":
                self.representation_[rep] = self.get_contrastive_keyword_layers(
                    n_keyphrases_per_cluster=self.n_keyphrases_per_cluster
                )
            else:
                warnings.warn(f"{rep} is not a supported representation")
        return None

    def build_base_prompt(
        self,
        cluster_id,
        layer_id=0,
        max_docs_per_cluster=100,
        max_adjacent_clusters=3,
        max_adjacent_docs=2,
        llm_instruction="The short distinguising topic name is:",
    ):
        """
        Take a cluster_id and layer_id and extracts the relevant information from the representation_ and cluster_layers_ properties to
        construct a representative prompt to present to a large langauge model.

        Each represenative is trimmed to be at most self.token_trim_length tokens in size.
        """
        prompt_text = f"Below is a information about a group of {self.document_type} from {self.corpus_description}:\n\n"

        # TODO: Add some random sentences

        # Add some contrastive keywords (might drop this in favor of the last one. Let the experiments commence!)
        if "contrastive" in self.representation_techniques:
            prompt_text += (
                'Distinguishing keywords for this group:\n - "'
                + ", ".join(self.representation_["contrastive"][layer_id][cluster_id])
                + '"\n'
            )
        # Add some topical documents
        if "topical" in self.representation_techniques:
            prompt_text += (
                f"\nSample topical {self.document_type} from the group include:\n"
            )
            for text in self.representation_["topical"][layer_id][cluster_id][
                :max_docs_per_cluster
            ]:
                prompt_text += f' - "{self._trim_text(text)}"\n'
            # Grab some of the same docs from nearby clusters for context.
            prompt_text += f"\n\nSimilar {self.document_type} from different groups with distinct topics include:\n"
            for adjacent_cluster_index in self.cluster_layers_.layer_cluster_neighbours[
                layer_id
            ][cluster_id][1:max_adjacent_clusters]:
                for text in self.representation_["topical"][layer_id][
                    adjacent_cluster_index
                ][:max_adjacent_docs]:
                    prompt_text += f'- "{self._trim_text(text)}"\n'
        # Add some documents from nearby clusters for contrast
        if "distinctive" in self.representation_techniques:
            prompt_text += (
                f"\nSample distinctive {self.document_type} from the group include:\n"
            )
            for text in self.representation_["distinctive"][layer_id][cluster_id][
                :max_docs_per_cluster
            ]:
                prompt_text += f' - "{self._trim_text(text)}"\n'
            prompt_text += f"\n\nSimilar {self.document_type} from different groups with distinct topics include:\n"
            # for adjacent_cluster_index in self.cluster_layers_.layer_cluster_neighbours[layer_id][cluster_id][1:max_adjacent_clusters]:
            #     for text in self.representation_['distinctive'][layer_id][adjacent_cluster_index][:max_adjacent_docs]:
            #         prompt_text += f"- \"{text}\"\n"
        prompt_text += "\n\n" + llm_instruction
        return prompt_text

    def fit_base_level_prompts(
        self,
        layer_id=0,
        max_docs_per_cluster=100,
        max_adjacent_clusters=3,
        max_adjacent_docs=2,
    ):
        """
        This returns a list of prompts for the layer_id independent of any other layer.
        This is commonly used for the base layer of a hierarchical topic clustering (hence the layer_id=0)

        If any of the prompt lengths (in llm tokenze) are longere than the max tokens for our llm (as defined by llm.n_ctx)
        then we reduce the maximum documents sampled from each cluster by a half and try again.  If we ever have to sample
        a single document per cluster we will declaire failure and raise and error.

        If the representation_ have not yet been generated or is None it will generate them as necessary.

        FUTURE: We hope to include improved subsampling and document partitioning method in future releases to allow
            for more representative sampling and prompt engineering.
        """
        max_docs_per_cluster = min(
            max_docs_per_cluster, self.n_sentence_examples_per_cluster
        )
        if self.verbose:
            print(
                f"generating base layer topic names with at most {max_docs_per_cluster} {self.document_type} per cluster."
            )
        if getattr(self, "representation_", None) is None:
            self.fit_representation()
        layer_size = len(self.cluster_layers_.location_layers[layer_id])
        prompts = []
        for cluster_id in tqdm(
            range(layer_size),
            desc="Generating base layer prompts",
            disable=(not self.verbose),
        ):
            prompt = self.build_base_prompt(
                cluster_id,
                layer_id,
                max_docs_per_cluster,
                max_adjacent_clusters,
                max_adjacent_docs,
                llm_instruction=self.llm.llm_instruction(kind="base_layer"),
            )
            prompt_length = len(self.llm.tokenize(prompt))
            reduced_docs_per_cluster = max_docs_per_cluster
            while prompt_length > self.llm.n_ctx():
                reduced_docs_per_cluster = reduced_docs_per_cluster // 2
                prompt = self.build_base_prompt(
                    cluster_id,
                    layer_id,
                    reduced_docs_per_cluster,
                    max_adjacent_clusters,
                    max_adjacent_docs,
                    llm_instruction=self.llm.llm_instruction(kind="base_layer"),
                )
                prompt_length = len(self.llm.tokenize(prompt))
                if reduced_docs_per_cluster < 1:
                    warnings.warn(
                        f"A prompt was too long for the context window and was trimmed: {prompt_length}> {self.llm.n_ctx()}"
                    )
                    prompt = self._trim_text(prompt, self.llm, self.llm.n_ctx())
            prompts.append(prompt)
        prompt_lengths = [len(self.llm.tokenize(prompt)) for prompt in prompts]
        self.base_layer_prompts_ = prompts
        return None

    def _get_topic_name(self, prompt_layer, layer_num):
        """
        Takes a prompt layer and applies an llm to convert these prompts into topics.
        """
        topic_names = []
        for i in tqdm(
            range(len(prompt_layer)),
            desc=f"Generating topics for layer {layer_num}",
            disable=(not self.verbose),
        ):
            topic_name = self.llm.generate_topic_name(prompt_layer[i])
            topic_names.append(topic_name)
        return topic_names

    def fit_base_layer_topics(self):
        """
        Uses the llm to fit a topic name for each base level cluster based on the base_layer_prompts_
        If the base_layer_prompts_ have not yet been generated or is None it will generate them as necessary.
        """
        if getattr(self, "base_layer_prompts_", None) is None:
            self.fit_base_level_prompts()
        self.base_layer_topics_ = self._get_topic_name(self.base_layer_prompts_, 0)
        return None

    def _topical_subtopics_for_cluster(
        self,
        layer_num,
        cluster_num,
        n_subtopics=32,
    ):
        metacluster = self.cluster_layers_.metacluster_layers[layer_num][cluster_num]
        pointset = self.cluster_layers_.pointset_layers[layer_num][cluster_num]

        centroid_vector = np.mean(self.document_vectors[pointset], axis=0)
        subtopic_vectors = np.asarray(
            [
                np.mean(
                    self.document_vectors[self.cluster_layers_.pointset_layers[0][n]],
                    axis=0,
                )
                for n in metacluster
            ]
        )
        candidate_neighbor_indices = np.argsort(
            np.squeeze(
                sklearn.metrics.pairwise_distances(
                    [centroid_vector], subtopic_vectors, metric="cosine"
                )
            )
        )[: 2 * n_subtopics]
        candidate_neighbors = subtopic_vectors[candidate_neighbor_indices]
        topical_subtopic_indices = candidate_neighbor_indices[
            diversify(
                centroid_vector,
                candidate_neighbors,
                alpha=0.66,
                max_candidates=n_subtopics,
            )
        ][:n_subtopics]
        topical_subtopics = [
            self.base_layer_topics_[metacluster[i]] for i in topical_subtopic_indices
        ]
        return topical_subtopics

    def _contrastive_subtopics_for_cluster(
        self,
        layer_num,
        cluster_num,
        base_layer_topic_embeddings,
        n_subtopics=24,
    ):

        layer_neighbors = self.cluster_layers_.layer_cluster_neighbours[layer_num]
        cluster_neighbors = layer_neighbors[cluster_num]
        meta_clusters = self.cluster_layers_.metacluster_layers[layer_num]
        cluster_subclusters = meta_clusters[cluster_neighbors[0]]
        topic_names = [self.base_layer_topics_[x] for x in cluster_subclusters]
        local_vectors = base_layer_topic_embeddings[
            sum([meta_clusters[x] for x in cluster_neighbors], [])
        ]
        U, S, Vh = np.linalg.svd(local_vectors - local_vectors.mean(axis=0))
        transformed_docs = local_vectors @ Vh.T
        transformed_docs = np.where(transformed_docs > 0, transformed_docs, 0)
        class_labels = np.repeat(
            np.arange(len(cluster_neighbors)),
            [len(meta_clusters[x]) for x in cluster_neighbors],
        )
        iwt = vectorizers.transformers.InformationWeightTransformer().fit(
            transformed_docs, class_labels
        )
        topic_name_weights = np.sum(
            transformed_docs[: len(topic_names)] * iwt.information_weights_, axis=1
        )
        distinctive_topic_indices = np.argsort(topic_name_weights)[: n_subtopics * 3]
        distinctive_topic_vectors = base_layer_topic_embeddings[
            distinctive_topic_indices
        ]
        diversified_candidates = diversify(
            base_layer_topic_embeddings[cluster_subclusters].mean(axis=0),
            distinctive_topic_vectors,
        )
        distinctive_topic_indices = distinctive_topic_indices[
            diversified_candidates[:n_subtopics]
        ]
        distinctive_sentences = [topic_names[i] for i in distinctive_topic_indices]
        return distinctive_sentences

    def fit_subtopic_layers(self, max_subtopics_per_cluster=32):
        """
        Fits the topical and contrastive subtopics for each intermadiate topic.
        If the base_layer_topics_ have not yet been generated or is None it will generate them as necessary.
        """
        if getattr(self, "base_layer_topics_", None) is None:
            self.fit_base_layer_topics()
        base_layer_topic_embedding = self.embedding_model.encode(
            self.base_layer_topics_, show_progress_bar=True
        )
        self.subtopic_layers_ = dict()
        self.subtopic_layers_["topical"] = [
            [
                self._topical_subtopics_for_cluster(
                    layer_num,
                    cluster_num,
                    n_subtopics=max_subtopics_per_cluster,
                )
                for cluster_num in range(
                    len(self.cluster_layers_.metacluster_layers[layer_num])
                )
            ]
            for layer_num in tqdm(
                range(1, len(self.cluster_layers_.metacluster_layers)),
                desc="Finding topical subtopics",
                disable=(not self.verbose),
            )
        ]
        self.subtopic_layers_["contrastive"] = [
            [
                self._contrastive_subtopics_for_cluster(
                    layer_num,
                    cluster_num,
                    base_layer_topic_embedding,
                    n_subtopics=max_subtopics_per_cluster,
                )
                for cluster_num in range(
                    len(self.cluster_layers_.metacluster_layers[layer_num])
                )
            ]
            for layer_num in tqdm(
                range(1, len(self.cluster_layers_.metacluster_layers)),
                desc="Finding contrastive subtopics",
                disable=(not self.verbose),
            )
        ]
        return None

    def _create_prompt_from_subtopics(
        self,
        previous_layer_topics,  # Need to find the previous layer topic that contained each topic.
        layer_id,
        max_subtopics=24,
        max_docs_per_cluster=4,
        max_adjacent_clusters=3,
        max_adjacent_docs=2,
        llm_instruction="The short distinguising topic name is:",
    ):
        if getattr(self, "subtopic_layers_", None) is None:
            self.fit_subtopic_layers(self.max_subtopics_per_cluster)
        layer_size = len(self.cluster_layers_.location_layers[layer_id])
        prompts = []
        for cluster_id in tqdm(
            range(layer_size),
            desc=f"Generating prompts for layer {layer_id}",
            disable=(not self.verbose),
        ):
            prompt_text = f"--\n\nBelow is a information about a group of {self.document_type} from {self.corpus_description} that are all on the same topic:\n\n"
            # Add some contrastive keywords
            if "contrastive" in self.representation_techniques:
                prompt_text += (
                    'Distinguishing keywords for this group:\n - "'
                    + ", ".join(
                        self.representation_["contrastive"][layer_id][cluster_id]
                    )
                    + '"\n'
                )
            # Use the previous layer information to inject knowledge into this cluster.
            prompt_text += "Sample sub-topics from the group include:\n"
            for text in previous_layer_topics[cluster_id][:max_subtopics]:
                prompt_text += f'- "{text}"\n'
            # Add some topical documents
            if "topical" in self.representation_techniques:
                prompt_text += (
                    f"\nSample topical {self.document_type} from the group include:\n"
                )
                for text in self.representation_["topical"][layer_id][cluster_id][
                    :max_docs_per_cluster
                ]:
                    prompt_text += f' - "{text}"\n'
                # Grab some of the same docs from nearby clusters for context.
                prompt_text += f"\n\nSimilar {self.document_type} from different groups with distinct topics include:\n"
                for (
                    adjacent_cluster_index
                ) in self.cluster_layers_.layer_cluster_neighbours[layer_id][
                    cluster_id
                ][
                    :max_adjacent_clusters
                ]:
                    for text in self.representation_["topical"][layer_id][
                        adjacent_cluster_index
                    ][:max_adjacent_docs]:
                        prompt_text += f'- "{text}"\n'
            # Add some distinctive keywords from this cluster and adjacent ones
            if "distinctive" in self.representation_techniques:
                prompt_text += f"\nSample distinctive {self.document_type} from the group include:\n"
                for text in self.representation_["distinctive"][layer_id][cluster_id][
                    :max_docs_per_cluster
                ]:
                    prompt_text += f' - "{text}"\n'

            prompt_text += "\nSub-topics from different but similar groups include:\n"
            for adjacent_cluster_index in self.cluster_layers_.layer_cluster_neighbours[
                layer_id
            ][cluster_id][:max_adjacent_clusters]:
                for text in previous_layer_topics[adjacent_cluster_index][
                    1:max_subtopics
                ]:
                    prompt_text += f'- "{text}"\n'
            prompt_text += "\n\n" + llm_instruction
            prompts.append(prompt_text)
        return prompts

    def fit_layers(self):
        """
        Constructs prompts and topic names for intermediate subtopic layers.
        If the subtopic_layers_ have not yet been generated or is None it will generate them as necessary.
        """

        if getattr(self, "subtopic_layers_", None) is None:
            self.fit_subtopic_layers(self.max_subtopics_per_cluster)
        if self.verbose:
            print(f"fitting intermediate layers")
        self.topic_prompt_layers_ = [self.base_layer_prompts_]
        self.topic_name_layers_ = [self.base_layer_topics_]
        # if(self.verbose):
        #     print(self.topic_name_layers_)

        for layer_id in range(1, len(self.cluster_layers_.metacluster_layers)):
            subtopics_layer = [
                list(zip(a, b))
                for a, b in zip(
                    self.subtopic_layers_["topical"][layer_id - 1],
                    self.subtopic_layers_["contrastive"][layer_id - 1],
                )
            ]
            topic_naming_prompts = self._create_prompt_from_subtopics(
                subtopics_layer,
                layer_id,
                llm_instruction=self.llm.llm_instruction(kind="intermediate_layer"),
            )
            self.topic_prompt_layers_.append(topic_naming_prompts)
            topic_names = self._get_topic_name(topic_naming_prompts, layer_id)
            self.topic_name_layers_.append(topic_names)
        return None

    def clean_topic_names(self):
        """
        Cleans up the prompts from the top down in order to remove topic name duplication.
        If previous properties have not yet been generated will generate them as necessary.
        This can be the only function called.
        """
        if getattr(self, "topic_name_layers_", None) is None:
            self.fit_layers()
        if self.verbose:
            print(f"cleaning up topic names\n")
        self.layer_clusters = [
            np.full(self.document_map.shape[0], "Unlabelled", dtype=object)
            for i in range(len(self.topic_name_layers_))
        ]
        unique_names = set(
            [""]
        )  # Start with empty string so we fix any topics that failed to get a name
        for n in range(len(self.topic_name_layers_) - 1, -1, -1):
            for i, (name, indices) in tqdm(
                enumerate(
                    zip(
                        self.topic_name_layers_[n],
                        self.cluster_layers_.pointset_layers[n],
                    )
                ),
                total=len(self.topic_name_layers_[n]),
                desc=f"Cleaning layer topic names for layer {n}",
                disable=(not self.verbose),
            ):
                n_attempts = 0
                unique_name = name
                original_topic_names = [unique_name]
                while unique_name in unique_names and n_attempts < 8:
                    prompt_text = create_final_remedy_prompt(
                        original_topic_names,
                        self.documents,
                        self.document_vectors,
                        indices,
                        self.cluster_layers_.vector_layers[n][i],
                        self.document_type,
                        self.corpus_description,
                        self.llm.llm_instruction(kind="remedy"),
                    )
                    unique_name = self.llm.generate_topic_name(prompt_text)
                    original_topic_names.append(unique_name)
                    n_attempts += 1
                if n_attempts > 0 and self.verbose:
                    print(f"{name} --> {unique_name} after {n_attempts} attempts")
                unique_names.add(unique_name)
                self.layer_clusters[n][indices] = unique_name

from utils.basic_utils import get_relevant_chunks, remove
import logging
from configuration import *
from operator import add
import time
from functools import reduce as reduce
from collections import Counter
from nltk.corpus import stopwords as sw
STOPWORDS = set(sw.words("english"))
from nltk.tokenize import word_tokenize as wt
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils.basic_utils import get_overlap
import copy
from .decorators.decorators import timed, memory_profile


def go_clean(doc):
    """
    Simple cleaning function
    :param doc: string (para or sentences)
    :return: tokenized words
    """
    doc = doc.lower()
    tokens = wt(doc)

    filterWord = []
    for w in tokens:
        if w not in PUNC and w not in STOPWORDS:
            if w in SLANGS:
                w = SLANGS[w]
            filterWord.append(w)

    sents = " ".join(filterWord)
    def deEmojify(inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')
    sents = deEmojify(sents)
    return sents


def get_inverted_index(docs):
    tokesize = lambda x: wt(x)
    tokens_array = list(map(tokesize, docs))

    inverted_index = {}
    _ = [[inverted_index.setdefault(word, []).append(i) \
          for word in tokens_array[i]] \
         for i in range(len(tokens_array))]
    return inverted_index


def get_preprocessing(data):
    """
    :param data: json object heaving headlines
    :return: data frames, and inverted index for further process
    """
    clean_data = [go_clean(i) for i in data]
    inverted_index = get_inverted_index(list(clean_data))

    return clean_data, inverted_index


def get_sents_vector(sents, model):
    """
    Function to convert given sentence into vectors
    :param sents: tokenized sentence
    :param model: model
    :return: sentence vectors
    """
    return vectorize_sentence(wt(sents), model)


def vectorize_sentence(sentence, model):
    """
    Function to aggregate all the word vectors to form a sentence vector
    :param sentence: List of word vectors
    :param model: ConoceptNet Embedding model
    :return: A aggregated sentence vector
    """
    final_vec = np.zeros((300, 1))
    if len(sentence):
        count = 0
    else:
        count = 1
    for word in sentence:
        count += 1
        dummy_vec = np.zeros((300, 1))
        try:
            temp_vec = get_vector(word, model)
            final_vec += temp_vec
        except:
            final_vec += dummy_vec
    return final_vec / count


def get_vector(word, model):
    """Function to fetch the vector for a word the model of pre-trained vectors."""
    return np.array(model.wv[word]).reshape((300, 1))


def get_similarity_score(clean_data, vectors, inverted_index, lam, default_dist=1):
    """

    :param Useful_df: data frames with headlines, cleaned headlines
    :param vectors: sentence vectors of hedlined
    :param inverted_index: inverted index of all headlines
    :param lam: combining paramiters of distance score
    :param default_dist: if word overlap is 0 than assume consin similarity =  default_dist
    :return: matrix maned score, where score[i][j] is the distance between i, j
    """

    score = word_overlap_score(inverted_index, clean_data, vectors, lam, default_dist)
    score = np.where(np.isnan(score), 1, score)

    return score


def word_overlap_score(inverted_index, docs, vectors, lam, default_dist=1):
    doc_len = len(docs)

    score_mat = np.full((doc_len, doc_len), default_dist, dtype="float32")

    index_array = lambda sents: reduce(lambda accu, indx_array: accu + indx_array,
                                       [inverted_index.get(word, []) for word in sents], [])

    tokens = [wt(sents.replace('.', '')) for sents in docs]
    words_inverded_index = list(map(index_array, tokens))
    count_dict_list = list(map(Counter, words_inverded_index))

    def get_score(i, j, intersect):
        n1 = len(tokens[i])
        n2 = len(tokens[j])

        overlap = 1 - 2 * float(intersect) / max(n1 + n2, 1)

        cosine = cosine_similarity(vectors[i].T, vectors[j].T)
        final_score = max((1 - lam) * overlap + lam * cosine.all(), 0)
        score_mat[i][j] = final_score
        score_mat[j][i] = final_score

    [[get_score(i, j, count_dict_list[i][j]) \
      for j in count_dict_list[i] if j >= i] \
     for i in tqdm(range(doc_len))]

    return score_mat


def get_hierarchical_clustering_label(distance_mat, cluster_start, cluster_end, step, source):
    """
    Function to do hierarchical clustering
    :param distance_mat: distance matrix of all questions of size len(questions)*len(questions)
    :param start: minimum number of cluster to be formed(starting range of number of clusters)
    :param end: maximum number of cluster to be formed (end range of number of clusters)
    :param step: number of steps to take each time for evaluating the clusters
    :return: Cluster labels of the best cluster set
    """
    if source == 'news':
        cluster_start = newscluster_startingpoint
        cluster_end = newscluster_endpoint
    else:
        cluster_start = twittercluster_startingpoint
        cluster_end = twittercluster_endpoint

    np.set_printoptions(precision=5, suppress=True)
    dist_condensed = []
    [dist_condensed.extend(distance_mat[i][i + 1:]) for i in range(len(distance_mat))]

    Z = linkage(np.array(dist_condensed), "average")  # creating linkage matrix
    scores = []
    labels = []
    for n_cluster in range(cluster_start, cluster_end, step):
        label = fcluster(Z, n_cluster, criterion='maxclust')
        sil_coeff = silhouette_score(distance_mat, label, metric='precomputed')
        logging.info(f'Cluster = {n_cluster} with silhoutte score {sil_coeff}')
        scores.append(sil_coeff)
        labels.append(label)

    m = max(scores)
    logging.info(f'max_score = {m} with cluster = {scores.index(m) + cluster_start}')
    return labels[scores.index(m)], scores.index(m) + cluster_start + 1


@timed
@memory_profile
def get_twitter_clusters(objects, model, prev_data):
    '''
    Fn to get twitter clusters
    :param objects: list of articles
    :param model: word2vec model
    :param prev_data: prev data list
    :return: clustered data
        '''

    tweet_objs = [tweet_obj for article in objects for tweet_obj in article['articles']]
    tweets = [obj.get('content') for obj in tweet_objs]
    queries = [obj.get('query_name') for obj in tweet_objs]
    logging.info(f'Got queries = {queries}')
    logging.info("Starting Hierarchical Clustering...")
    logging.info(f'Total no of tweets got = {len(tweets)}')

    clean_data, inverted_index = get_preprocessing(tweets)
    logging.debug(f"----Articles cleaned.----")

    vectors = [get_sents_vector(data, model) for data in clean_data]

    tic = time.time()
    score = get_similarity_score(clean_data, vectors, inverted_index, SIMILARITY_LAM,
                                 default_dist=DEFAULT_SIMILARITY_SCORE)
    print(f'time taken in getting matrix = {time.time() - tic}')

    logging.debug("----Distance matrix created----")

    labels, n_clusters = get_hierarchical_clustering_label(score, int(len(clean_data)*CLUSTER_START_PERCENTAGE), int(len(clean_data)*CLUSTER_END_PERCENTAGE), CLUSTER_STEP, 'twitter')
    logging.debug("----Hierarchical Clustering completed successfully----")
    labels = list(labels)

    indexes = [get_indexes(n, labels) for n in range(1, n_clusters)]
    cluster_data = [{'Label': n, 'count': labels.count(n), 'samples': get_data(n, labels, tweet_objs),
                    'representatives': get_representative(idx, vectors, get_data(n, labels, tweet_objs))}
                     for n, idx in zip(range(1, n_clusters), indexes)]
    if len(prev_data):
        logging.info('-------Merging of Clusters Started-----------')
        res = merge_clusters(prev_data, cluster_data, model, 'twitter')
        return res
    else:
        return cluster_data


@timed
@memory_profile
def get_news_clusters(objects, model, prev_data):
    '''
    Fn to get news clusters
    :param objects: list of articles
    :param model: word2vec model
    :param prev_data: prev data list
    :return: clustered data
    '''

    chunks = [[get_relevant_chunks(obj['query'], article['content'], article.get('_id', ''),
                                   article.get('created', ''), article.get('source_url', ''))
               for article in obj['articles']] for obj in objects]
    chunks = [sampleofsample for chunk in chunks for sample in chunk for sampleofsample in sample
              if len(sampleofsample) > 0]                # flatting the list

    articles_chunks = remove(chunks)
    chunks = [article.get('text', '') for article in articles_chunks]
    logging.info(f'Total no of chunks formed = {len(chunks)}')
    logging.info("Starting Hierarchical Clustering...")

    if len(chunks):
        clean_data, inverted_index = get_preprocessing(chunks)
        logging.debug(f"----Articles cleaned.----")

        vectors = [get_sents_vector(data, model) for data in clean_data]

        tic = time.time()
        score = get_similarity_score(clean_data, vectors, inverted_index, SIMILARITY_LAM,
                                     default_dist=DEFAULT_SIMILARITY_SCORE)
        print(f'time taken in getting matrix = {time.time() - tic}')

        logging.debug("----Distance matrix created----")

        labels, n_clusters = get_hierarchical_clustering_label(score, int(len(clean_data)*CLUSTER_START_PERCENTAGE), int(len(clean_data)*CLUSTER_END_PERCENTAGE), CLUSTER_STEP, 'news')
        logging.debug("----Hierarchical Clustering completed successfully----")
        labels = list(labels)
        indexes = [get_indexes(n, labels) for n in range(1, n_clusters)]
        cluster_data = [{'Label': n, 'count': labels.count(n), 'samples': get_data(n, labels, articles_chunks),
                        'representatives': get_representative(idx, vectors, get_data(n, labels, articles_chunks))}
                        for n, idx in zip(range(1, n_clusters), indexes)]

        if len(prev_data):
            logging.info('-------Merging of Clusters Started-----------')
            res = merge_clusters(prev_data, cluster_data, model, 'news')
            logging.info('-------Clusters Merging Done-----------')
            return res
        else:
            return cluster_data


def get_representative(index, embs, articles):
    try:
        new_embs = [np.array(embs[idx]).reshape(300, 1) for idx in index]
        centroid = np.mean(new_embs, axis=0)
        sims = [float(cosine_similarity(emb.T, centroid.T).reshape(1, -1)) for emb in new_embs]
        original_sims = copy.deepcopy(sims)
        sims.sort(reverse=True)
        similar_indexes = sims[:n_representative]
        indexes = [original_sims.index(n) for n in similar_indexes]
        represents = [articles[idx] for idx in indexes]
        logging.info('------Extraction of Representatives done------')
        return represents
    except Exception as ex:
        logging.exception(ex)
        return []


@timed
@memory_profile
def merge_clusters(prev_data, new_data, model, source):
    '''
    Fn to merge new clusters with prev clusters
    :param prev_data: list of prev data with prev labels
    :param new_data: list of new data with new labels
    :param model: word2vec model
    :param source: news/ twitter
    :return: new data after merging with prev data
    '''
    try:
        prev_labels = [prev.get('label') for prev in prev_data]
        max_label = max(prev_labels)
        prev_list = [i.get('representatives') for i in prev_data]
        for new in new_data:
            if source == 'twitter':
                rep_list = [rep.get('content', '') for rep in new.get('representatives')]
            else:
                rep_list = [rep.get('text', '') for rep in new.get('representatives')]
            cosine_sim = list(map(lambda sent:
                                  float(cosine_similarity(get_sents_vector(' '.join(rep_list), model).T,
                                                          get_sents_vector(' '.join([s.get('content', '') for s in sent]), model).T)),
                                  prev_list))

            overlap = list(map(lambda sent:
                               float(get_overlap(' '.join(rep_list), ' '.join([s.get('content', '') for s in sent]))),
                               prev_list))

            score = list(map(add, (1 - SIMILARITY_LAM) * np.array(overlap), SIMILARITY_LAM * np.array(cosine_sim)))
            indexes = []
            for idx, (s, prev) in enumerate(zip(score, prev_data)):
                if s >= merging_threshold:
                    logging.debug(f'Merging cluster found with score {s} new label {new["Label"]} updated to prev label {prev["label"]}')
                    new['Label'] = prev['label']
                    break
                else:
                    indexes.append(idx)
            if len(indexes) != 0:
                logging.debug(f'new label {new["Label"]} updated to {max_label+1}')
                new['Label'] = max_label + 1
                max_label = max_label + 1
        return new_data

    except Exception as ex:
        logging.error('Merging Failed')
        logging.exception(ex)
        return new_data


def get_data(n, labels, chunks):
    return [chunks[idx] for idx, label in enumerate(labels) if label == n]


def get_indexes(n, labels):
    return [idx for idx, label in enumerate(labels) if label == n]

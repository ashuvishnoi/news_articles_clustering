# paths
AWS = True
if AWS:
    EMBEDDING_MODEL_PATH = '/home/ubuntu/pretrain_files/news_suspecto_w2v'
else:
    EMBEDDING_MODEL_PATH = 'C:\\Users\\Ashutosh\\Desktop\\Unfound\\pretrain_files\\5_million_sentences'

# EMBEDDING_MODEL_PATH = 'C:\\Users\\Ashutosh\\Desktop\\Unfound\\pretrain_files\\5_million_sentences'
maxenvlen = 2


CLUSTER_START_PERCENTAGE = 0.05
CLUSTER_END_PERCENTAGE = 0.10
newscluster_startingpoint = 3
newscluster_endpoint = 30
CLUSTER_STEP = 1
SIMILARITY_LAM = 0.4
DEFAULT_SIMILARITY_SCORE = 1
n_representative = 2
merging_threshold = 0.6

twittercluster_startingpoint = 3
twittercluster_endpoint = 100

PUNC = ['.', ',', '!', '', "+", "#", "(", ")", ":", "'s", "'", '"', '@']
SLANGS = {"n't": "not", "r": "are", "u": "you"}

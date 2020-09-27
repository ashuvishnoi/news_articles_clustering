import logging
logger = logging.getLogger(__name__)
from app.clustering_core import get_news_clusters, get_twitter_clusters
from utils.text_utils import *


def articles_clustering_service(input_obj, model):
    new_data = input_obj.get('new_data')
    prev_data_news = input_obj.get('prev_data', {}).get('news')
    prev_data_twitter = input_obj.get('prev_data', {}).get('twitter')

    try:
        tweets_data = [article for article in new_data if article.get('source') == 'twitter']
        news_data = [article for article in new_data if article.get('source') == 'news']
        try:
            tweets_clusters = get_twitter_clusters(tweets_data, model, prev_data_twitter)
            logging.info('---------Twitter clustering done successfully------------')
        except Exception as ex:
            logging.exception(ex)
            tweets_clusters = []
        try:
            news_clusters = get_news_clusters(news_data, model, prev_data_news)
            logging.info('---------News clustering done successfully------------')
        except Exception as ex:
            logging.exception(ex)
            news_clusters = []
        obj = {'twitter_clusters': tweets_clusters, 'news_clusters': news_clusters}
        response = create_response_object(len(obj), obj, STATUS_SUCCESS)
        return response

    except Exception as ex:
        logger.exception("Error: {}".format(ex))
        response = create_response_object(STATUS_FAILED, [], DATA_EXTRACTION_FAILED)
        return response


def create_response_object(len_data, data, message):
    """
    Function to create a standard response object for sending back to request.
    """
    if len_data != 0:
        response = {'status': STATUS_SUCCESS, 'message': message,
                    'result': data}
        logger.info(message)
        return response

    else:
        response = {'status': STATUS_FAILED, 'message': DATA_EXTRACTION_FAILED,
                    'result': {}}
        logger.info(DATA_EXTRACTION_FAILED)
        return response


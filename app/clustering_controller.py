from flask import Blueprint, request, jsonify
import time
import logging
from utils.text_utils import BAD_INPUT, ARTICLES_CLUSTERING_ENDPOINT
from app.clustering_service import articles_clustering_service
from utils.basic_utils import exception_response
logger = logging.getLogger(__name__)
articles_clustering_api = Blueprint('articles_clustering', __name__)


def clustering_handler(model):
    @articles_clustering_api.route(ARTICLES_CLUSTERING_ENDPOINT, methods=['POST'])
    def clustering_controller():
        try:
            if request.is_json:
                tic = time.time()
                response = articles_clustering_service(request.get_json(), model)

                result = jsonify(response['result'])
                toc = time.time()
                logger.info(f'TIME TAKEN TO GET CLUSTERED DATA: {toc - tic} Secs')
                return result

            else:
                return jsonify(BAD_INPUT)

        except Exception as ex:
            return exception_response(ex)

    return articles_clustering_api

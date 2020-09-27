from flask_cors import CORS
from flask import Flask, jsonify
import logging
from flask_swagger import swagger
from app.clustering_controller import clustering_handler
import os
from configuration import EMBEDDING_MODEL_PATH
from gensim.models import Word2Vec


# Create a custom logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# SETUP LOGGING
if not os.path.exists('logs'):
    os.makedirs('logs')


# Create handlers
debug_handler = logging.FileHandler('logs/debug.log')
info_handler = logging.FileHandler('logs/info.log')
debug_handler.setLevel(logging.DEBUG)
info_handler.setLevel(logging.INFO)


def load_word2vec_model():
    return Word2Vec.load(EMBEDDING_MODEL_PATH)


model = load_word2vec_model()
logging.info('-----------------Word2Vec Loaded Successfully----------------')

articles_clustering_api = clustering_handler(model)
CORS(articles_clustering_api)

# Start the flask server
server = Flask(__name__)
server.register_blueprint(articles_clustering_api)


@server.route("/", methods=['POST', 'GET'])
def spec():
    return jsonify(swagger(server))
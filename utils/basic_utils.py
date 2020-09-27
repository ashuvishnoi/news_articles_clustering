import logging
from utils.text_utils import ERROR_EXCEPTION_OCCUR, STATUS_FAILED
from flask import jsonify
logger = logging.getLogger(__name__)
from configuration import maxenvlen
from nltk import sent_tokenize
from nltk import word_tokenize as wt


# show response when exception occur
def exception_response(ex):
    message = ERROR_EXCEPTION_OCCUR.format(type(ex).__name__, ex.args)
    logger.error(message, exc_info=True)
    response = error_response(message)
    return jsonify(response)


# show response for errors
def error_response(msg, result=None):
    logger.error(msg)
    response = {'status': STATUS_FAILED, 'message': msg, 'result': result}
    return response


def remove(duplicate):
    final_list = []
    final_list1 =[]
    for num in duplicate:
        if num.get('text') not in final_list:
            final_list.append(num.get('text'))
            final_list1.append(num)
    return final_list1


def get_relevant_chunks(query, text, id, date, url):
    logging.debug('Get Relevant Chunks from text Started')
    logging.debug(f'Got query = {query}')
    sents = sent_tokenize(text)
    query1 = query.lower().split()
    query2 = query.split()
    chunks = ['']
    for q1, q2 in zip(query1, query2):
        if q1 in text or q2 in text:
            chunks = get_chunks(q1, q2, sents)
            chunks = [' '.join(data.replace(query, '').split()) for data in chunks]
            # chunks = [data for data in chunks]
            break
    res = [{'_id': id, 'date': date, 'text': chunk, 'source_url': url} for chunk in chunks]
    return res


def get_chunks(q1, q2, sents):
    try:
        idx = 0
        chunks = []
        while idx <= len(sents) - 1:

            if (q1 in sents[idx] or q2 in sents[idx]) and len(sents) > maxenvlen:
                if idx == 0:
                    chunk = ' '.join([sents[i] for i in range(0, maxenvlen)])
                    idx = idx + maxenvlen
                elif idx == len(sents) - 1:
                    chunk = ' '.join([sents[i] for i in range(len(sents) - maxenvlen, len(sents))])
                    idx = idx + len(sents)
                else:
                    key = maxenvlen / 2
                    if maxenvlen % 2 != 0 and idx + int(key + 0.5) <= len(sents):
                        chunk = ' '.join(
                            [sents[i] for i in range(idx - int(key - 0.5), idx + int(key + 0.5))])
                        idx = idx + int(key + 0.5)
                    elif idx + int(key) <= len(sents):
                        chunk = ' '.join([sents[i] for i in range(idx - int(key), idx + int(key))])
                        idx = idx + int(key)
                    else:
                        num = len(sents) - 1 - idx
                        upper = idx + num
                        lower = idx - int(key) - (int(key) - num)
                        chunk = ' '.join([sents[i] for i in range(lower, upper)])
                        idx = idx + lower

                chunks.append(chunk)
            else:
                idx += 1
        if len(chunks) == 0:
            chunks = ['']
        return chunks

    except:
        logging.info('Exception as Index gone out of Range and Returning empty list')
        return ['']


def get_overlap(sent1, sent2):
    sent1 = set(wt(sent1))
    sent2 = set(wt(sent2))
    try:
        value = max(len(sent1.intersection(sent2)) / len(sent1.union(sent2)), 0)
        return value
    except:
        return 0

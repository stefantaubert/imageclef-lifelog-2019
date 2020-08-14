import string

from nltk import word_tokenize
from nltk.corpus import stopwords

from src.globals import top_desc, top_narrative, top_title
from src.query_translation.get_query import get_query

#nltk.download('stopwords')

__stopwords__ = set(stopwords.words('english'))

__query_typical_tokens__ = [
    "find",
    "moment",
    "moments",
    "u1",
    "u2",
]

def remove_stopwords(words: list):
    return [w for w in words if w.lower() not in __stopwords__]

def remove_query_typical_words(words: list):
    return [w for w in words if w.lower() not in __query_typical_tokens__]

def remove_punctuation(sentence: str):
    return sentence.translate(str.maketrans('', '', string.punctuation))

def remove_duplicates(words: list):
    return list(set(words))

def remove_not_known_words(words: list, vocab: set):
    return [w for w in words if w in vocab]

def get_query_tokens(query: str, vocab: set) -> list:
    desc = remove_punctuation(query)
    desc = desc.lower()
    words = word_tokenize(desc)
    words = remove_stopwords(words)
    words = remove_query_typical_words(words)
    words = remove_duplicates(words)
    words = remove_not_known_words(words, vocab)
    words = sorted(words)
    return words

def get_query_tokens_from_topic(topic: dict, query_src: str, ds: int, vocab: set) -> list:
    query = get_query(topic, query_src, ds)
    return get_query_tokens(query, vocab)

def get_query_tokens_from_topics(topics: list, query_src: str, ds: int, vocab: set) -> list:
    query_tokens = [get_query_tokens_from_topic(topic, query_src, ds, vocab) for topic in topics]
    return query_tokens

# if __name__ == "__main__":
#     from src.globals import ds_dev
#     from src.globals import ds_test
#     from src.globals import top_desc
#     from src.models.pooling.Model_opts import *
#     

#     dev_opts = {
#         opt_general: {
#             opt_usr: 1,
#             
#             opt_ds: ds_dev,
#         },
        
#         opt_model: {
#             opt_query_src: query_src_desc,
#         }
#     }

#     test_opts = {
#         opt_general: {
#             opt_usr: 1,
#             
#             opt_ds: ds_test,
#         },
        
#         opt_model: {
#             opt_query_src: query_src_desc,
#         }
#     }

#     dev_ctx = Context(dev_opts)
#     test_ctx = Context(test_opts)
#     all_tops = []
#     for top in dev_ctx.tops(): all_tops.append(top)
#     for top in test_ctx.tops(): all_tops.append(top)

#     for top in all_tops:
#         print(top[top_desc])
#         print(get_query_tokens_from_topic(top, dev_ctx))
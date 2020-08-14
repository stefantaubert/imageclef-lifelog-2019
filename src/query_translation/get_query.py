from src.globals import top_desc
from src.globals import top_title
from src.globals import top_narrative
from src.globals import top_idi
from src.globals import ds_dev
from src.globals import ds_test
from src.query_translation.get_query_opts import *
from src.io.ReadingContext import ReadingContext

manual_query_table_dev = {
    1: "eating, sea, icecream",
    2: "eating, drinking, restaurant, food",
    3: "watching, video",
    4: "taking, bridge, photo",
    5: "shopping, grocery shop, food",
    6: "playing guitar",
    7: "cooking, food",
    8: "car sales showroom",
    9: "taking, public transportation",
    10: "reading, paper, book",
}

manual_query_table_test = {
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
    7: "",
    8: "",
    9: "",
    10: "",
}

# manual_query_table_dev = {
#     1: "sea",
#     2: "restaurant",
#     3: "video",
#     4: "bridge",
#     5: "grocery",
#     6: "guitar",
#     7: "cooking",
#     8: "showroom",
#     9: "transportation",
#     10: "paper",
# }

# manual_query_table_test = {
#     1: "toyshop",
#     2: "driving",
#     3: "fridge",
#     4: "watching",
#     5: "coffee",
#     6: "breakfast",
#     7: "cafe",
#     8: "smartphone",
#     9: "shirt",
#     10: "meeting",
# }

def get_query(topic: dict, query_src: str, ds: int) -> str:
    if query_src == query_src_desc:
        return topic[top_desc]
    elif query_src == query_src_title:
        return topic[top_title]
    elif query_src == query_src_narrative:
        return topic[top_narrative]
    elif query_src == query_src_manual:
        if ds == ds_dev:
            return manual_query_table_dev[topic[top_idi]]
        elif ds == ds_test:
            return manual_query_table_test[topic[top_idi]]
        else: assert False
    else: assert False
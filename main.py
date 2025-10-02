import os

from config import load_config
from treesearch.search import TreeSearch
from utils.log import set_log_level


def main():
    set_log_level(os.getenv("ISGSA_LOG", "INFO"))

    config = load_config()

    # TODO:
    user_request = (
        "Which of these LensKit algorithms performs the best on MovieLens100K?"
        "- ItemItem"
        "- BiasedMF"
        "I placed the 'u.data' file of MovieLens100K in your current working directory. You can load it from there!"
    )

    ts = TreeSearch(user_request, config=config)
    ts.run()


if __name__ == "__main__":
    main()

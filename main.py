import os

from config import load_config
from treesearch.search import TreeSearch
from utils.log import set_log_level


def main():
    set_log_level(os.getenv("ISGSA_LOG", "INFO"))

    config = load_config()

    # TODO:
    task_desc = "Some task description lorem ipsum dolor"

    ts = TreeSearch(task_desc, config)
    ts.run()


if __name__ == "__main__":
    main()

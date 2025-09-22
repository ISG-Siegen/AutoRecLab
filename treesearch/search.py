from config import Config

# from treesearch.minimal_agent import MinimalAgent
from treesearch.node import Node
from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("treesearch")


class TreeSearch:
    def __init__(self, task_desc: str, config: Config) -> None:
        self._task_desc = task_desc
        self._config = config
        self._draft_nodes: list[Node] = []

    def select_next_nodes(self) -> Node:
        # TODO:
        raise NotImplementedError()

    def run(self):
        # minimal_agent = MinimalAgent(self._task_desc, self._config)

        # Step 1: Generate draft nodes:
        for i in range(self._config.treesearch.num_draft_nodes):
            logger.info(i)

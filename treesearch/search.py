import random
from pathlib import Path

from marshmallow import pre_dump

from config import Config

# from treesearch.minimal_agent import MinimalAgent
from treesearch.interpreter import Interpreter
from treesearch.minimal_agent import MinimalAgent
from treesearch.node import Node
from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("treesearch")


class TreeSearch:
    def __init__(self, user_request: str, config: Config) -> None:
        self._user_request = user_request
        self._config = config
        self._draft_nodes: list[Node] = []
        workspace_pth = Path(config.exec.workspace).resolve()
        workspace_pth.mkdir(exist_ok=True, parents=True)
        self._workspace = str(workspace_pth)

        self._minimal_agent = MinimalAgent(self._task_desc, self._config)
        self._interpreter = Interpreter(self._workspace, self._config.exec.timeout)

    @property
    def all_nodes(self):
        all_nodes = self._draft_nodes
        for draft_node in self._draft_nodes:
            all_nodes.extend(draft_node.descendants)

        return all_nodes

    @property
    def good_nodes(self):
        return list(filter(lambda n: not n.is_buggy, self.all_nodes))

    @property
    def buggy_nodes(self):
        return list(filter(lambda n: n.is_buggy, self.all_nodes))

    @property
    def best_good_node(self):
        good_nodes = self.good_nodes
        good_nodes.sort(key=lambda n: n.score.overall_score, reverse=True)
        return good_nodes[0]

    def select_next_node(self) -> Node:
        if random.random() < self._config.treesearch.debug_prob:
            return random.choice(self.buggy_nodes)

        # Epsilon-greedy explore vs. exploit:
        if random.random() < self._config.treesearch.epsilon:
            return random.choice(self.good_nodes)
        else:
            return self.best_good_node

    def run(self):
        # Step 1: Generate draft nodes:
        for i in range(self._config.treesearch.num_draft_nodes):
            logger.info(
                f"Generating draft node {i}/{self._config.treesearch.num_draft_nodes}"
            )
            draft_node = self._minimal_agent._draft()
            self.exec_node(draft_node)

        for i in range(self._config.treesearch.max_iterations):
            logger.info(
                f"Treesearch iteration {i}/{self._config.treesearch.max_iterations}"
            )
            parent_node = self.select_next_node()

            if parent_node.is_buggy:
                child_node = self._minimal_agent._debug(parent_node)
            else:
                child_node = self._minimal_agent._improve(parent_node)

            self.exec_node(child_node)

            if child_node.score.is_satisfactory:
                logger.info("Found satisfactory node:")
                # HACK:
                print(child_node.term_out)
                print(child_node.code)
                return

        logger.warning("Found no satisfactory node; Using best node instead...")

        best_node = self.best_good_node

        print(best_node.term_out)
        print(best_node.code)

    def exec_node(self, node: Node) -> Node:
        exec_result = self._interpreter.run(node.code)
        self._minimal_agent.score_code(node, exec_result)
        return node

    @property
    def _task_desc(self) -> str:
        task_desc = """ You are an expert recommender systems research assistant who is looking to help the user with their requests.
The user has some idea and you want to conduct creative experiments to gain scientific insights.
Your aim is to run experiments to gather sufficient results to report back to the user.
The idea is:\n
"""
        task_desc += self._user_request
        return task_desc

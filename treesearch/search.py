from pathlib import Path

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

    def select_next_nodes(self) -> Node:
        # TODO:
        raise NotImplementedError()

    def run(self):
        minimal_agent = MinimalAgent(self._task_desc, self._config)

        interpreter = Interpreter(self._workspace, self._config.exec.timeout)

        # Step 1: Generate draft nodes:
        for _ in range(self._config.treesearch.num_draft_nodes):
            draft_node = minimal_agent._draft()
            exec_result = interpreter.run(draft_node.code)
            minimal_agent.parse_exec_result(draft_node, exec_result, self._workspace)

        """ TODO:
        - [x] Execute/eval subroutine on draft nodes:
            - [x] Run code
            - [ ] Score node
        - [ ] Select node
        - [ ] If buggy: debug
        - [ ] Otherwise improve
        - [ ] Call the execute/eval subroutine
        """

    @property
    def _task_desc(self):
        task_desc = """ You are an expert recommender systems research assistant who is looking to help the user with their requests.
The user has some idea and you want to conduct creative experiments to gain scientific insights.
Your aim is to run experiments to gather sufficient results to report back to the user.
The idea is:\n
"""
        task_desc += self._user_request
        return task_desc

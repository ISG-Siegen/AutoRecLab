import tomllib
from pathlib import Path

import tomli_w
from pydantic_settings import BaseSettings

CONFIG_PATH = Path("./config.toml")


class TreeSearchConfig(BaseSettings):
    num_draft_nodes: int = 3
    debug_prob: float = 0.5
    epsilon: float = 0.3
    max_iterations: int = 10
    satisfactory_threshold: float = 7.5


class ExecConfig(BaseSettings):
    timeout: int = 3600
    workspace: str = "./workspace"


class CodeConfig(BaseSettings):
    model: str = "gpt-4o"
    model_temp: float = 1.0


class AgentConfig(BaseSettings):
    k_fold_validation: int = 1
    data_preview: bool = False
    code: CodeConfig = CodeConfig()


class Config(BaseSettings):
    treesearch: TreeSearchConfig = TreeSearchConfig()
    exec: ExecConfig = ExecConfig()
    agent: AgentConfig = AgentConfig()


def load_config():
    if not CONFIG_PATH.exists():
        config = Config()
        with CONFIG_PATH.open("wb") as conf_fh:
            tomli_w.dump(config.model_dump(), conf_fh)
        return config

    with CONFIG_PATH.open("rb") as conf_fh:
        data = tomllib.load(conf_fh)
        return Config(**data)

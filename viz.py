import pickle
import subprocess
from pathlib import Path
from tempfile import TemporaryFile

from anytree.exporter.dotexporter import UniqueDotExporter

from treesearch.node import Node


def main():
    output_dir = Path("./tree_render")

    with open("./save.pkl", "rb") as f:
        draft_nodes: list[Node] = pickle.load(f)

        tmp_file = output_dir / "tmp.dot"

        for idx, dn in enumerate(draft_nodes):
            e = UniqueDotExporter(dn)
            e.to_dotfile(tmp_file)

            out_file = output_dir / f"tree{idx}.png"

            subprocess.run(["dot", str(tmp_file), "-T", "png", "-o", str(out_file)])


if __name__ == "__main__":
    main()

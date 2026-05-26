import pickle
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from anytree import Node as AnyNode
from anytree.exporter.dotexporter import UniqueDotExporter

from treesearch.node import Node
from utils.path import mkdir


def _score_color(node: Node) -> tuple[str, str, str]:
    if getattr(node, "is_buggy", False):
        return "#FEF2F2", "#EF4444", "#991B1B"

    score = getattr(node.score, "score", 0.0)

    if getattr(node.score, "is_satisfactory", False) or score >= 0.75:
        return "#F0FDF4", "#22C55E", "#166534"
    elif score >= 0.5:
        return "#EFF6FF", "#3B82F6", "#1E40AF"
    elif score >= 0.25:
        return "#FEFCE8", "#EAB308", "#854D0E"
    else:
        return "#F8FAFC", "#94A3B8", "#1E293B"


def _short_id(node_id: str) -> str:
    if len(node_id) <= 8:
        return node_id
    return f"{node_id[:4]}..{node_id[-4:]}"


def _node_attr(node: Node) -> str:
    if getattr(node, "is_virtual_root", False):
        return 'style="invis" label="" width="0" height="0"'

    fill, border, font = _score_color(node)
    score_val = getattr(node.score, "score", 0.0)
    satisfactory = getattr(node.score, "is_satisfactory", False)

    if satisfactory:
        status = f"✓ satisfactory  {score_val:.3f}"
    elif getattr(node, "is_buggy", False):
        status = f"✗ buggy  {score_val:.3f}"
    else:
        status = f"score: {score_val:.3f}"

    type_info = ""
    if getattr(node, "type_check_passed", None) is not None:
        attempts = getattr(node, "type_check_attempts", 1)
        passed = "✓" if node.type_check_passed else "✗"
        type_info = f"\\ntype-check: {passed} ({attempts} att.)"

    label = f"{_short_id(node.id)}\\n{status}{type_info}"

    return (
        f'label="{label}" '
        f'style="filled,rounded" '
        f'fillcolor="{fill}" '
        f'color="{border}" '
        f'fontcolor="{font}" '
        f'fontname="Inter,Helvetica Neue,Arial,sans-serif" '
        f'fontsize="10" '
        f'shape="box" '
        f'margin="0.25,0.15" '
        f'penwidth="1.2" '
    )


def _edge_attr(node: Node, child: Node) -> str:
    if getattr(node, "is_virtual_root", False):
        return 'style="invis"'

    _, border, _ = _score_color(child)
    return (
        f'color="{border}" '
        f'penwidth="1.2" '
        f'arrowsize="0.6" '
        f'arrowhead="vee" '
    )


GRAPH_OPTIONS = [
    'bgcolor="#FFFFFF"',
    'pad="0.4"',
    'nodesep="0.6"',
    'ranksep="0.5"',
    'splines="true"',
    'rankdir="TB"',
]


def render_trees(nodes: list[Node], output_dir: Path):
    if not nodes:
        return

    virtual_root = AnyNode("VIRTUAL_ROOT")
    virtual_root.is_virtual_root = True

    for root in nodes:
        root.parent = virtual_root

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_file = tmp_dir / "tmp.dot"

        e = UniqueDotExporter(
            virtual_root,
            nodeattrfunc=_node_attr,
            edgeattrfunc=_edge_attr,
            options=GRAPH_OPTIONS,
        )
        e.to_dotfile(tmp_file)

        for fmt in ["png", "pdf", "svg"]:
            fmt_dir = mkdir(output_dir / fmt)
            out_file = fmt_dir / f"combined_tree.{fmt}"

            cmd = ["dot", str(tmp_file), f"-T{fmt}", "-o", str(out_file)]
            if fmt == "png":
                cmd.append("-Gdpi=192")

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[combined] dot error ({fmt}): {result.stderr.strip()}")
            else:
                print(f"[combined] → {out_file}")

    for root in nodes:
        root.parent = None


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-pkl", default="./save.pkl")
    parser.add_argument("-o", "--output-dir", default="./tree_render")
    args = parser.parse_args()

    if not Path(args.input_pkl).exists():
        print(f'Input file "{args.input_pkl}" does not exist!')
        return

    output_dir = mkdir(args.output_dir)

    with open(args.input_pkl, "rb") as f:
        nodes: list[Node] = pickle.load(f)
        render_trees(nodes, output_dir)


if __name__ == "__main__":
    main()
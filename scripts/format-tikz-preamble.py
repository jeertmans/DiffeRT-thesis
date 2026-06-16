import argparse
import re
from pathlib import Path


def file_path(path) -> Path:
    path = Path(path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File '{path}' does not exist.")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path '{path}' is not a file.")
    return path


def folder_path(path) -> Path:
    path = Path(path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Folder '{path}' does not exist.")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path '{path}' is not a directory.")
    return path


def format_tikz_preamble(tikz_file: Path):
    with open(tikz_file, "r") as f:
        lines = f.readlines()

    req_regex = re.compile(
        r"(?P<usepackage>\\usepackage{.*})|(?:\\usetikzlibrary{(?P<usetikzlibrary>.*)})|(?:\\usepgfplotslibrary{(?P<usepgfplotslibrary>.*)})|(?P<pgfplotsset>\\pgfplotsset{.*})|(?P<providecommand>\\providecommand{.*})|(?P<include>INCLUDE .* relto .*)"
    )

    requires = False
    pkg_reqs = []
    tikz_reqs = set()
    pgfplots_reqs = set()
    pgfplotsset = None
    commands = []
    includes = []

    nonreq_lines = set()

    for i, line in enumerate(lines):
        if line.startswith("% Requires") or line.startswith(
            "% No specific requirements"
        ):
            pass
        elif line.startswith("%") and (req := req_regex.search(line)):
            m_group = req.lastgroup
            m = req.group(m_group)
            match m_group:
                case "usepackage":
                    pkg_reqs.append(line)
                case "usetikzlibrary":
                    tikz_reqs.update(r.strip() for r in m.split(","))
                case "usepgfplotslibrary":
                    pgfplots_reqs.update(r.strip() for r in m.split(","))
                case "pgfplotsset":
                    if pgfplotsset is not None:
                        raise ValueError(
                            f"Multiple pgfplotsset found in {tikz_file}: '{pgfplotsset}' and '{m}'"
                        )
                    pgfplotsset = line
                case "providecommand":
                    commands.append(line)
                case "include":
                    includes.append(line)

        else:
            nonreq_lines.add(i)

    if (
        len(pkg_reqs) == 0
        and len(tikz_reqs) == 0
        and len(pgfplots_reqs) == 0
        and pgfplotsset is None
        and len(commands) == 0
        and len(includes) == 0
    ):
        requires = False
    else:
        requires = True

    formatted_lines = ["% Requires:\n" if requires else "% No specific requirements\n"]
    formatted_lines += pkg_reqs
    formatted_lines += [f"% \\usetikzlibrary{{{lib}}}\n" for lib in sorted(tikz_reqs)]
    formatted_lines += [
        f"% \\usepgfplotslibrary{{{lib}}}\n" for lib in sorted(pgfplots_reqs)
    ]
    if pgfplotsset is not None:
        formatted_lines.append(pgfplotsset)
    formatted_lines += commands
    formatted_lines += includes
    formatted_lines += [lines[i] for i, line in enumerate(lines) if i in nonreq_lines]

    with open(tikz_file, "w") as f:
        f.writelines(formatted_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format TikZ files preamble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tikz-folder",
        help="TikZ files folder",
        type=folder_path,
        default=folder_path(Path(__file__).parent.parent / "tikz"),
    )
    args = parser.parse_args()

    # Format preamble for all TikZ files
    for tikz_file in args.tikz_folder.glob("*.tex"):
        format_tikz_preamble(tikz_file)

import argparse
import re
from collections import defaultdict
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize and merge TikZ files dependencies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--main",
        help="main TeX file",
        type=file_path,
        default=file_path(Path(__file__).parent.parent / "main.tex"),
    )
    parser.add_argument(
        "--tikz-folder",
        help="TikZ files folder",
        type=folder_path,
        default=folder_path(Path(__file__).parent.parent / "tikz"),
    )
    parser.add_argument(
        "--include-libraries",
        help="include TikZ libraries in the preamble, otherwise print them",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    pgfplotslibraries = defaultdict(set)
    tikzlibraries = defaultdict(set)

    tikz_req_regex = re.compile(r"\\usetikzlibrary{(.*)}")
    pgfplots_req_regex = re.compile(r"\\usepgfplotslibrary{(.*)}")

    # Gather requirements from all TikZ files
    for tikz_file in args.tikz_folder.glob("*.tex"):
        with open(tikz_file, "r") as f:
            for line in f.readlines():
                if line.startswith("%") and (req := tikz_req_regex.search(line)):
                    for library in req.group(1).split(","):
                        library = library.strip()
                        tikzlibraries[library].add(tikz_file.name)
                if line.startswith("%") and (req := pgfplots_req_regex.search(line)):
                    for library in req.group(1).split(","):
                        library = library.strip()
                        pgfplotslibraries[library].add(tikz_file.name)

    if not args.include_libraries:
        print("Required PGFPlots libraries:")
        for library in sorted(pgfplotslibraries.keys()):
            files = sorted(pgfplotslibraries[library])
            print(f"- {library}: from files {', '.join(files)}")
        print("Required TikZ libraries:")
        for library in sorted(tikzlibraries.keys()):
            files = sorted(tikzlibraries[library])
            print(f"- {library}: from files {', '.join(files)}")
    else:
        start_index = end_index = None
        with open(args.main, "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                if line.startswith(r"% START: PGFPlots/TikZ libraries"):
                    start_index = i + 1
                    while len(lines) > i + 1 and (
                        not lines[i + 1].startswith(r"% END: PGFPlots/TikZ libraries")
                    ):
                        i += 1
                    end_index = i
                    break

        if start_index is not None and end_index is not None:
            with open(args.main, "w") as f:
                f.writelines(lines[:start_index])
                for library in sorted(pgfplotslibraries.keys()):
                    files = sorted(pgfplotslibraries[library])
                    f.write(
                        f"\\usepgfplotslibrary{{{library}}} % Automatically imported from files {', '.join(files)}\n"
                    )

                for library in sorted(tikzlibraries.keys()):
                    files = sorted(tikzlibraries[library])
                    f.write(
                        f"\\usetikzlibrary{{{library}}} % Automatically imported from files {', '.join(files)}\n"
                    )
                f.writelines(lines[end_index + 1 :])
        else:
            print("No PGFPlots/TikZ libraries section found in the main TeX file.")

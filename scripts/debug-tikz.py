# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "psutil",
#     "watchfiles",
# ]
# ///
import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import psutil
from watchfiles import watch


def file_path(path) -> Path:
    path = Path(path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File '{path}' does not exist.")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path '{path}' is not a file.")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug script for TikZ figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="input TikZ file", type=file_path)
    parser.add_argument("--cmd", help="command to run", type=str, default="pdflatex")
    parser.add_argument(
        "--doc",
        help="document class and option",
        type=str,
        default="[crop,tikz]{standalone}",
    )
    parser.add_argument(
        "--open",
        help="open the output PDF",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--open-cmd", help="command to open the PDF", type=str, default="xdg-open"
    )
    parser.add_argument(
        "--watch",
        help="watch for changes and recompile",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--verbose", help="show compilation output", default=False, action="store_true"
    )
    parser.add_argument(
        "--cleanup",
        help="remove temporary files (including output PDF) when done",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    req_regex = re.compile(
        r"(\\usepackage(?:\[[^\]]*\])?{.*})|(\\usetikzlibrary{.*})|(\\usepgfplotslibrary{.*})|(\\pgfplotsset{.*})|(\\providecommand{.*})"
    )
    inc_regex = re.compile(r"INCLUDE (.*) relto (.*)")

    with tempfile.TemporaryDirectory(
        prefix=f"tikz_debug_{args.input.stem}_", delete=args.cleanup
    ) as tmp_dir:
        main_tex = Path(tmp_dir) / "main.tex"

        def compile_pdf() -> subprocess.Popen:
            with open(args.input, "r") as tikz_file, open(main_tex, "w") as main_file:
                main_file.write(f"\\documentclass{args.doc}\n")
                reqs = []
                for line in tikz_file.readlines():
                    if line.startswith("%"):
                        if req := req_regex.search(line):
                            reqs.append(req.group(0))
                        if inc := inc_regex.search(line):
                            inc_file = (args.input.parent / inc.group(1)).resolve()
                            dst_file = Path(tmp_dir) / inc_file.relative_to(
                                (args.input.parent / inc.group(2)).resolve().parent
                            )
                            dst_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(inc_file, dst_file)
                main_file.writelines(reqs)
                main_file.writelines(
                    [
                        "\\begin{document}",
                        "\\input{" + str(args.input.resolve()) + "}",
                        "\\end{document}",
                    ]
                )

            subprocess.run(
                [*shlex.split(args.cmd), "main.tex"],
                cwd=tmp_dir,
                stdout=sys.stdout if args.verbose else subprocess.PIPE,
                stderr=sys.stderr,
            )

        compile_pdf()
        out_file = str((Path(tmp_dir) / "main.pdf").resolve())
        if args.open:
            p = subprocess.Popen(
                [*shlex.split(args.open_cmd), out_file],
                stdout=sys.stdout if args.verbose else subprocess.PIPE,
                stderr=sys.stderr,
            )
        if args.watch:
            print(f"Watching {args.input} for changes...")
            for _ in watch(args.input, raise_interrupt=False):
                print(f"Change detected in {args.input}, recompiling...")
                compile_pdf()

        p.wait()

        # If the command opened a separate process, wait for it to finish
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            if out_file in p.info["cmdline"]:
                p.wait()

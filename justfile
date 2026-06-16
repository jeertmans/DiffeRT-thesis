# Default command (list all commands)
default:
  just --list

alias b := build

# Compile document
build:
  latexmk

# Compile once
build-once:
  latexmk -e '$max_repeat=1'

alias c := clean

# Clean build artifacts
clean:
  latexmk -CA

# Clean build artifacts and TikZ cache
clean-all: clean
  find tikzexternalize ! -name '.gitkeep' -type f -exec rm -f {} +

# Format source files and run checks
fmt:
  pre-commit run --all-files

# Install necessary dependencies (requires Just and Python installed)
install:
  pip install pre-commit  # Install pre-commit

# Update dependencies
update:
  pre-commit autoupdate

# Compile document, open it, and recompile on changes
watch:
  latexmk -pvc

# Driftwood User Guide

Driftwood is a small command-line tool for keeping plain-text notes in a folder and syncing them between computers. Driftwood is a made-up program written for this RAG tutorial, so Claude cannot know anything about it from its training data. Every correct answer about Driftwood has to come from this document.

## Installation

Driftwood needs Python 3.11 or newer. The installer creates a `drift` command and a notes folder at `~/driftwood` the first time it runs.

### Linux

On Debian and Ubuntu, install the package with `sudo apt install driftwood`. On Fedora, use `sudo dnf install driftwood`. If your distribution does not package Driftwood, install it from PyPI with `pipx install driftwood`, which keeps it isolated from your system Python. After installing, run `drift init` once to create the notes folder and the default configuration file.

### macOS

On macOS, install Driftwood with Homebrew by running `brew install driftwood`. Apple Silicon and Intel Macs are both supported. The first time you run `drift`, macOS may ask for permission to access your Documents folder if you moved your notes there.

### Windows

Windows is supported through WSL only. Install WSL with Ubuntu, then follow the Linux instructions inside the WSL terminal. Native Windows builds are not available because Driftwood relies on POSIX file locking.

## Configuration

Driftwood reads its settings from `~/.config/driftwood/config.toml`. The most important setting is `notes_dir`, which controls where notes are stored. The default editor is taken from the `EDITOR` environment variable; set `editor = "nvim"` in the config file to override it. The `date_format` setting controls the timestamp at the top of each new note and defaults to `%Y-%m-%d`.

Notes are plain markdown files. Driftwood never changes a note's contents on its own; it only reads the files and the small index it keeps in `.driftwood/index.db`.

## Writing notes

Run `drift new "Meeting with Sam"` to create a note. Driftwood turns the title into a file name such as `2026-03-14-meeting-with-sam.md` and opens it in your editor. Use `drift list` to show recent notes and `drift search kayak` to find every note containing the word kayak. Tags are written as `#tag` anywhere in a note, and `drift tags` lists all tags with their counts.

## Syncing

Driftwood syncs notes through any folder that another tool already keeps in sync, such as Syncthing, Dropbox, or a git repository. Set `sync_dir` in the config file to that folder, then run `drift sync`. Sync is two-way: new and changed notes are copied in both directions.

When the same note was edited on two computers before syncing, Driftwood does not try to merge the changes. It keeps your local version and saves the other version next to it with a `.conflict` suffix, for example `2026-03-14-meeting-with-sam.conflict.md`. Resolve the conflict by editing the files by hand and deleting the `.conflict` copy. Run `drift sync --dry-run` to see what would change without copying anything.

## Troubleshooting

If `drift` reports "index is locked", another Driftwood process is still running, or a previous one crashed. Close other terminals running Driftwood and try again. If the error remains, delete `.driftwood/index.lock` inside your notes folder.

If search results look out of date, rebuild the index with `drift reindex`. Rebuilding takes a few seconds for a few thousand notes and never modifies the notes themselves.

To report a bug, run `drift doctor` and include its output. It prints your Driftwood version, Python version, config file location, and the number of notes found.

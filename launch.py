"""Launch SGLang server with beam search plugin.

Usage:
    python launch.py --model-path <model> --port 8000 [other sglang args...]
"""

import os
import sys

from sglang_beamsearch import load_beam_search_plugin

# Apply monkeypatches BEFORE sglang starts
load_beam_search_plugin()

from sglang.srt.server_args import prepare_server_args
from sglang.launch_server import run_server
from sglang.srt.utils import kill_process_tree

server_args = prepare_server_args(sys.argv[1:])

try:
    run_server(server_args)
finally:
    kill_process_tree(os.getpid(), include_parent=False)

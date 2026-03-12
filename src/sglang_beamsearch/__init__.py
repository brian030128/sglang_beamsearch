from sglang_beamsearch.plugin import apply_beam_search_patches


def load_beam_search_plugin():
    """Entry point: call this before launching the SGLang server to install beam search patches."""
    apply_beam_search_patches()

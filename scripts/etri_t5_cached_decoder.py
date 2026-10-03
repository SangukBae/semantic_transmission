"""Run tail17 reconstruction with prepared T5 tensors and no loaded T5 model."""
import os
from pathlib import Path
import runpy

from semantic_transmission.text_embedding_cache import install_cached_encoder


def main():
    repo = Path(__file__).resolve().parents[1]
    install_cached_encoder(Path(os.environ["ETRI_TEXT_EMBEDDINGS"]),
                           Path(os.environ["ETRI_TEXT_EMBEDDING_TRACE"]), repo)
    runpy.run_path(str(repo / "scripts/etri_tail17_decoder.py"), run_name="__main__")


if __name__ == "__main__":
    main()

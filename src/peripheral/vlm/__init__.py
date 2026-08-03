"""VLM backends. Phase 1: llama.cpp via the prebuilt llama-server binary over localhost HTTP."""

from .llama_server import LlamaServerClient, VlmResult, build_client

__all__ = ["LlamaServerClient", "VlmResult", "build_client"]

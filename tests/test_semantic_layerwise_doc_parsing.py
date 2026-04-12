import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


        
from typing import Callable, TypeVar, ParamSpec, cast
import time
from joblib import Memory
import pytest

P = ParamSpec("P")
R = TypeVar("R")

OLLAMA_SEMANTIC_MODELS = ["gemma4:e2b"]
GEMINI_SEMANTIC_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]
LOCAL_KOGWISTAR_BASE_URL = "http://127.0.0.1:28110"

def cached(memory: Memory, fn: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn))


class ProgressPrinter:
    def __init__(self, *, title: str, total: int) -> None:
        self.title = title
        self.total = total
        self.index = 0
        self.started_at = time.perf_counter()

    def banner(self) -> None:
        print(f"\n=== {self.title} ===", flush=True)

    def item(self, label: str) -> None:
        self.index += 1
        width = 18
        filled = 0 if self.total == 0 else max(0, min(width, round(width * self.index / self.total)))
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{self.index:>3}/{self.total:<3}] [{bar}] {label}", flush=True)

    def substage(self, label: str) -> None:
        print(f"    -> {label}", flush=True)

    def finish(self) -> None:
        elapsed = time.perf_counter() - self.started_at
        print(f"=== {self.title} done in {elapsed:.1f}s ===\n", flush=True)


def _configure_parser_env(monkeypatch: pytest.MonkeyPatch, *, provider: str, model: str, gemini_key: str | None = None) -> None:
    monkeypatch.setenv("KG_DOC_PARSER_PROVIDER", provider)
    monkeypatch.setenv("KG_DOC_PARSER_MODEL", model)
    if provider == "ollama":
        monkeypatch.setenv("KG_DOC_PARSER_BASE_URL", "http://127.0.0.1:11434")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    elif gemini_key:
        monkeypatch.setenv("GOOGLE_API_KEY", gemini_key)


def _wait_for_local_server(base_url: str = LOCAL_KOGWISTAR_BASE_URL, *, timeout_s: float = 20.0) -> None:
    import requests

    deadline = time.perf_counter() + timeout_s
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=1.5)
            if response.ok:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.2)
    pytest.skip(f"local Kogwistar server is not healthy at {base_url}: {last_error}")



def _load_version_chain_db():
    """Load the legacy VersionChainDB helper if it exists, otherwise skip.

    This legacy test file still depends on the old file-version-chain helper in
    some environments. If that helper is not present in the current checkout,
    the affected tests should be skipped rather than failing during import.
    """
    try:
        from src.utils.version_chaining import VersionChainDB
    except ModuleNotFoundError:
        pytest.skip("legacy VersionChainDB helper is not available in this checkout")
    return VersionChainDB

@pytest.mark.parametrize(
    "parser_provider,model_names",
    [
        pytest.param("ollama", OLLAMA_SEMANTIC_MODELS, id="ollama", marks=pytest.mark.ci_full),
        pytest.param("gemini", GEMINI_SEMANTIC_MODELS, id="gemini", marks=pytest.mark.manual),
    ],
)
def test_semantic_document_splitting(gemini_key, monkeypatch, parser_provider, model_names):
    """End-to-end smoke test for legacy semantic splitting over OCR split pages.

    Reads existing split-page OCR artifacts, rebuilds the semantic tree, round-
    trips it through the graph payload conversion, and exercises the downstream
    validation and index/search endpoints against a representative document set.

    The Ollama path is the CI-full default; the Gemini path is kept as a manual
    fallback for environments that still want to validate the legacy Google
    model ladder. The legacy semantic parser and index builders are cached by
    joblib under `.joblib/`; delete that directory for a fresh cacheless run,
    especially when retrying the manual Gemini case.

    The test prints a small progress banner, per-document progress rows, and
    per-stage substeps so a local run is easy to follow without digging through
    captured pytest output.
    """
    from src.utils.file_loaders import RawFileLoader, find_folders_two_levels_from_leaves_mem_optimized
    import os
    def filter_callback (file_path):
        # folder at least have some page ocr that ends with .json
        for i, f in enumerate(os.listdir(file_path)):
            if f.endswith('.json'):
                return True
        else:
            return False

    # memory = Memory(".cache", verbose=0)
    # compute_cached = cached(memory, compute)
    compare_root = os.path.join('..', 'doc_data', 'split_pages')
    loader = RawFileLoader(env_flist_path=None,
                           walk_root=os.path.join('..', 'doc_data', 'split_pages', ''),
                           compare_root = os.path.join('..', 'doc_data', 'split_pages'),
                           filtering_callbacks = [filter_callback],
                           include = ['dirs']
                           )
    from src.semantic_document_splitting_layerwise_edits import (parse_doc, 
                                                                 semantic_tree_to_kge_payload, 
                                                                 kge_payload_to_semantic_tree,
                                                                 build_index_terms_for_semantic_node,
                                                                 all_child_from_root,
                                                                 SemanticNode)
    from src.ocr import regen_doc
    from joblib import Memory
    import uuid, os
    memory = Memory(location = os.path.join('.joblib','test','test_semantic_document_splitting',str(parser_provider),str('__'.join(model_names))))
    if parser_provider == "gemini" and not gemini_key:
        pytest.skip("gemini_key is required for the manual Gemini semantic-splitting case")
    _configure_parser_env(
        monkeypatch,
        provider=parser_provider,
        model=model_names[0],
        gemini_key=gemini_key,
    )
    pytest.importorskip("langchain_ollama" if parser_provider == "ollama" else "langchain_google_genai")
    selected_docs = list(loader)
    progress = ProgressPrinter(title=f"semantic splitting [{parser_provider}]", total=len(selected_docs))
    progress.banner()
    for f in selected_docs:
        f_name = pathlib.Path(f).name
        progress.item(f_name)
        progress.substage("load raw OCR split page bundle")
        doc = {f_name: regen_doc(os.path.join(compare_root, f), use_raw=True)}

        progress.substage("parse document tree")
        document_tree, source_map = parse_doc(doc_id=f_name, raw_doc_dict=doc, model_names=model_names)

        progress.substage("serialize tree to graph payload")
        cached_semantic_tree_to_kge_payload = cached(memory, semantic_tree_to_kge_payload)
        graph_to_persist = cached_semantic_tree_to_kge_payload(document_tree)

        progress.substage("round-trip graph payload back to semantic tree")
        reconstrcted_root = kge_payload_to_semantic_tree(graph_to_persist)
        assert reconstrcted_root.model_dump() == document_tree.model_dump()

        _wait_for_local_server()
        import requests

        progress.substage("validate graph with local service")
        res = requests.post(f"{LOCAL_KOGWISTAR_BASE_URL}/api/document.validate_graph", json=graph_to_persist)
        res.raise_for_status()

        progress.substage("build index terms")
        all_child_nodes = all_child_from_root(reconstrcted_root)
        all_nodes = all_child_nodes + [reconstrcted_root]
        batch_index_list = build_index_terms_for_semantic_node(
            all_nodes,
            doc_id=f_name,
            model_names=model_names,
            mode="bottom_up_digest",
        )
        payload = {"index": [i.model_dump(mode="json") for i in batch_index_list]}
        for k in payload["index"]:
            k.update({"doc_id": str(reconstrcted_root.node_id)})

        @memory.cache
        def get_index_entries(payload):
            res = requests.post(f"{LOCAL_KOGWISTAR_BASE_URL}/api/add_index_entries", json=payload)
            return res

        progress.substage("push index entries")
        res = get_index_entries(payload)
        res.raise_for_status()

        import json

        def upsert_doc(doc, doc_id):
            res = requests.post(
                f"{LOCAL_KOGWISTAR_BASE_URL}/api/document",
                json={
                    "doc_id": doc_id,
                    "doc_type": "ocr_document",
                    "insertion_method": "document_parser_v1",
                    "content": doc,
                },
            )
            res.raise_for_status()
            return res

        progress.substage("upsert document record")
        res = upsert_doc(json.dumps(doc), str(reconstrcted_root.node_id))
        res.raise_for_status()

        def get_upsert_result(graph_to_persist):
            res = requests.post(f"{LOCAL_KOGWISTAR_BASE_URL}/api/document.upsert_tree", json=graph_to_persist)
            res.raise_for_status()
            return res

        progress.substage("upsert semantic tree")
        res = get_upsert_result(graph_to_persist)
        res.raise_for_status()

        progress.substage("search index and resolve node")
        res = requests.get(f"{LOCAL_KOGWISTAR_BASE_URL}/api/search_index_hybrid", params={"q": '"6.7"'})
        res.raise_for_status()
        res = requests.get(
            f"{LOCAL_KOGWISTAR_BASE_URL}/api/search_index_hybrid",
            params={"q": '"6.7"', "resolve_node": True},
        )
        res.raise_for_status()
    progress.finish()
    return
@pytest.mark.parametrize(
    "parser_provider,model_names",
    [
        pytest.param("ollama", OLLAMA_SEMANTIC_MODELS, id="ollama", marks=pytest.mark.ci_full),
        pytest.param("gemini", GEMINI_SEMANTIC_MODELS, id="gemini", marks=pytest.mark.manual),
    ],
)
def test_semantic_document_splitting_pdf_indexed(gemini_key, monkeypatch, parser_provider, model_names):
    """Smoke-test the legacy PDF-indexed flow with the same cached model split.

    Uses the version-chain database to decide which PDF files are eligible for
    splitting, then runs the PDF-to-page export path and semantic reconstruction
    against the selected PDFs. The Ollama path is the CI-full default; the
    Gemini path remains a manual fallback. The legacy parser/index builders are
    cached by joblib under `.joblib/`; delete that directory for a fresh
    cacheless rerun, especially when retrying the manual Gemini case.
    """
    from pdf2png import batch_split_pdf
    from src.utils.file_loaders import RawFileLoader
    import os
    from functools import lru_cache
    from joblib import Memory

    _configure_parser_env(
        monkeypatch,
        provider=parser_provider,
        model=model_names[0],
        gemini_key=gemini_key,
    )
    if parser_provider == "gemini" and not gemini_key:
        pytest.skip("gemini_key is required for the manual Gemini semantic-splitting case")
    pytest.importorskip("langchain_ollama" if parser_provider == "ollama" else "langchain_google_genai")

    VersionChainDB = _load_version_chain_db()
    file_index_root = (pathlib.Path(os.getcwd()).parent / "doc_data" / "file_version_chains").absolute()
    compare_root = os.path.join("..", "doc_data", "raw_documents")
    walk_root = os.path.join("..", "doc_data", "raw_documents", "Samples - 9 Oct 2025")

    loader = RawFileLoader(
        env_flist_path=None,
        walk_root=walk_root,
        compare_root=compare_root,
        include=["files"],
    )

    @lru_cache(maxsize=10)
    def get_chain_index(folder_path):
        rel_path = pathlib.Path(folder_path).relative_to(compare_root)
        db = VersionChainDB(str(pathlib.Path(file_index_root) / rel_path / "file_index.sqlite"))
        return db

    def filter_callback(file_path):
        folder_path = pathlib.Path(file_path).parent
        db = get_chain_index(folder_path)
        return db.is_canonical_by_name(file_path)

    selected_loader = RawFileLoader(
        env_flist_path=None,
        walk_root=walk_root,
        compare_root=compare_root,
        include=["files"],
        filtering_callbacks=[filter_callback],
    )

    splitted_folder = (pathlib.Path(os.getcwd()).parent / "doc_data" / "split_pages").absolute()
    batch_split_pdf(file_loader=selected_loader, outfolder_path=splitted_folder, exists_ok="skip")

    from semantic_document_splitting_layerwise_edits import (
        parse_doc,
        semantic_tree_to_kge_payload,
        kge_payload_to_semantic_tree,
        build_index_terms_for_semantic_node,
        all_child_from_root,
    )
    from src.ocr import regen_doc

    memory = Memory(location=".joblib")
    for f in selected_loader:
        f_name = pathlib.Path(f).name
        pdf_page_root = pathlib.Path(splitted_folder) / pathlib.Path(f)
        doc = regen_doc(str(pdf_page_root / f_name), use_raw=True)
        document_tree, source_map = parse_doc(doc_id=f_name, raw_doc_dict=doc, model_names=model_names)
        cached_semantic_tree_to_kge_payload = memory.cache(semantic_tree_to_kge_payload)
        graph_to_persist = cached_semantic_tree_to_kge_payload(document_tree)
        reconstrcted_root = kge_payload_to_semantic_tree(graph_to_persist)
        assert reconstrcted_root.model_dump() == document_tree.model_dump()
        _wait_for_local_server()
        import requests
        res = requests.post(f"{LOCAL_KOGWISTAR_BASE_URL}/api/document.validate_graph", json=graph_to_persist)
        res.raise_for_status()
        nodes = all_child_from_root(reconstrcted_root)
        batch_index_list = build_index_terms_for_semantic_node(nodes, model_names=model_names)
        payload = {"index": [i.model_dump(mode='json') for i in batch_index_list]}
        for k in payload["index"]:
            k.update({"doc_id": str(reconstrcted_root.node_id)})

        @memory.cache
        def get_index_entries(payload):
            res = requests.post(f"{LOCAL_KOGWISTAR_BASE_URL}/api/add_index_entries", json=payload)
            return res

        res = get_index_entries(payload)
        res.raise_for_status()

        @memory.cache
        def get_upsert_result(graph_to_persist):
            res = requests.post(f"{LOCAL_KOGWISTAR_BASE_URL}/api/document.upsert_tree", json=graph_to_persist)
            return res

        res = get_upsert_result(graph_to_persist)
        res.raise_for_status()
        res = requests.get(f"{LOCAL_KOGWISTAR_BASE_URL}/api/search_index_hybrid", params={"q": '"6.7"'})
        res.raise_for_status()
        res = requests.get(
            f"{LOCAL_KOGWISTAR_BASE_URL}/api/search_index_hybrid",
            params={"q": '"6.7"', "resolve_node": True},
        )
        res.raise_for_status()
        return
    
@pytest.mark.parametrize(
    "parser_provider,model_names",
    [
        pytest.param("ollama", OLLAMA_SEMANTIC_MODELS, id="ollama", marks=pytest.mark.ci_full),
        pytest.param("gemini", GEMINI_SEMANTIC_MODELS, id="gemini", marks=pytest.mark.manual),
    ],
)
def test_semantic_document_splitting_doc_group(gemini_key, monkeypatch, parser_provider, model_names):
    """Smoke-test grouped document parsing and indexing with cached model runs.

    Loads grouped raw documents, rebuilds the semantic tree, persists the graph,
    upserts the document and tree, and verifies the search/index flow still works
    for a representative document-group batch. The Ollama path is the CI-full
    default; the Gemini path remains a manual fallback. The legacy parser and
    index builders are joblib-cached under `.joblib/`; delete that directory
    for a fresh cacheless rerun, especially when retrying the manual Gemini
    case.
    """
    from src.utils.file_loaders import RawFileLoader
    import os
    from functools import lru_cache
    from joblib import Memory

    _configure_parser_env(
        monkeypatch,
        provider=parser_provider,
        model=model_names[0],
        gemini_key=gemini_key,
    )
    if parser_provider == "gemini" and not gemini_key:
        pytest.skip("gemini_key is required for the manual Gemini semantic-splitting case")
    pytest.importorskip("langchain_ollama" if parser_provider == "ollama" else "langchain_google_genai")

    file_index_root = (pathlib.Path(os.getcwd()).parent / "doc_data" / "file_version_chains").absolute()
    compare_root = os.path.join("..", "doc_data", "raw_documents")
    loader = RawFileLoader(
        env_flist_path=None,
        walk_root=os.path.join("..", "doc_data", "raw_documents", "Samples - 9 Oct 2025"),
        compare_root=compare_root,
        include=["files"],
    )

    from semantic_document_splitting_layerwise_edits import (
        parse_doc,
        semantic_tree_to_kge_payload,
        kge_payload_to_semantic_tree,
        build_index_terms_for_semantic_node,
        all_child_from_root,
    )
    from src.ocr import regen_doc_group

    memory = Memory(location=".joblib")
    VersionChainDB = _load_version_chain_db()

    @lru_cache(maxsize=10)
    def get_chain_index(folder_path):
        rel_path = pathlib.Path(folder_path).relative_to(compare_root)
        db = VersionChainDB(str(pathlib.Path(file_index_root) / rel_path / "file_index.sqlite"))
        return db

    def filter_callback(file_path):
        folder_path = pathlib.Path(file_path).parent
        db = get_chain_index(folder_path)
        return db.is_canonical_by_name(file_path)

    for f in loader:
        f_name = pathlib.Path(f).name
        doc_group = regen_doc_group(os.path.join(loader.compare_root, f), use_raw=True)
        document_tree, source_map = parse_doc(f_name, doc_group, model_names=model_names)
        cached_semantic_tree_to_kge_payload = memory.cache(semantic_tree_to_kge_payload)
        graph_to_persist = cached_semantic_tree_to_kge_payload(document_tree)
        reconstrcted_root = kge_payload_to_semantic_tree(graph_to_persist)
        assert reconstrcted_root.model_dump() == document_tree.model_dump()
        import requests
        res = requests.post("http://127.0.0.1:28110/api/document.validate_graph", json=graph_to_persist)
        res.raise_for_status()
        nodes = all_child_from_root(reconstrcted_root)
        batch_index_list = build_index_terms_for_semantic_node(nodes, model_names=model_names)
        payload = {"index": [i.model_dump(mode='json') for i in batch_index_list]}
        for k in payload["index"]:
            k.update({"doc_id": str(reconstrcted_root.node_id)})

        @memory.cache
        def get_index_entries(payload):
            res = requests.post("http://127.0.0.1:28110/api/add_index_entries", json=payload)
            return res

        res = get_index_entries(payload)
        res.raise_for_status()

        @memory.cache
        def get_upsert_result(graph_to_persist):
            res = requests.post("http://127.0.0.1:28110/api/document.upsert_tree", json=graph_to_persist)
            return res

        res = get_upsert_result(graph_to_persist)
        res.raise_for_status()
        res = requests.get("http://127.0.0.1:28110/api/search_index_hybrid", params={"q": '"6.7"'})
        res.raise_for_status()
        res = requests.get("http://127.0.0.1:28110/api/search_index_hybrid", params={"q": '"6.7"', "resolve_node": True})
        res.raise_for_status()
        return

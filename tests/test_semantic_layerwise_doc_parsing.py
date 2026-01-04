import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


        
from typing import Callable, TypeVar, ParamSpec, cast
from joblib import Memory

P = ParamSpec("P")
R = TypeVar("R")

def cached(memory: Memory, fn: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn))

def test_semantic_document_splitting(gemini_key):
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
                                                                 all_child_from_root)
    from src.ocr import regen_doc
    from joblib import Memory
    import uuid
    memory = Memory(location = '.joblib')
    for f in loader:
        f_name = pathlib.Path(f).name
        doc = {f_name : regen_doc(os.path.join(compare_root, f), use_raw = True)}
        document_tree, source_map = parse_doc(doc_id=f_name, raw_doc_dict=doc)
        cached_semantic_tree_to_kge_payload = cached(memory, semantic_tree_to_kge_payload)
        graph_to_persist = cached_semantic_tree_to_kge_payload(document_tree)
        reconstrcted_root = kge_payload_to_semantic_tree(graph_to_persist)
        assert reconstrcted_root.model_dump() == document_tree.model_dump()
        import requests
        res = requests.post("http://127.0.0.1:28110/api/document.validate_graph", json = graph_to_persist)
        res.raise_for_status()
        nodes = all_child_from_root(reconstrcted_root)
        batch_index_list = build_index_terms_for_semantic_node(nodes, doc_id=f_name)        
        payload = {"index": [i.model_dump(mode='json') for i in batch_index_list]}
        for k in payload['index']:
            k.update({'doc_id': str(reconstrcted_root.node_id)})
        @memory.cache
        def get_index_entries(payload):
            res = requests.post("http://127.0.0.1:28110/api/add_index_entries", json = payload)
            return res
        res = get_index_entries(payload)
        res.raise_for_status()
        @memory.cache
        def get_upsert_result(graph_to_persist):
            res = requests.post("http://127.0.0.1:28110/api/document.upsert_tree", json = graph_to_persist)
            return res
        res = get_upsert_result(graph_to_persist)
        res.raise_for_status()
        # search using index
        # sample visualization call http://localhost:28110/viz/d3?doc_id={document_tree.node_id}&mode=reify&insertion_method=document_parser_v1
        res = requests.get("http://127.0.0.1:28110/api/search_index_hybrid", 
                            params = {'q' : '"6.7"'})
        res.raise_for_status()
        res = requests.get("http://127.0.0.1:28110/api/search_index_hybrid", 
                            params = {'q' : '"6.7"', "resolve_node": True})
        res.raise_for_status()
        return
def test_semantic_document_splitting_pdf_indexed(gemini_key):
    from pdf2png import batch_split_pdf
    from src.utils.file_loaders import RawFileLoader#, find_folders_two_levels_from_leaves_mem_optimized
    import os
    from functools import lru_cache
    from src.utils.version_chaining import VersionChainDB
    file_index_root = (pathlib.Path(os.getcwd()).parent/"doc_data"/ "file_version_chains").absolute()
    @lru_cache(maxsize = 10)
    def get_chain_index(folder_path):
        rel_path = pathlib.Path(folder_path).relative_to(compare_root)
        try:
            db = VersionChainDB(str(pathlib.Path(file_index_root) / rel_path / "file_index.sqlite"))
        except Exception as e:
            raise e
        return db
    def filter_callback (file_path):
        folder_path = pathlib.Path(file_path).parent
        db = get_chain_index(folder_path)
        # stats = db.get_canonical_page_statistics()
        # print(f"{folder_path}: {stats=}")
        # for filestat in os.listdir(doc_group_folder_path):
        tf_use_file = db.is_canonical_by_name(file_path)
        return tf_use_file
            # else:
                # print('nono')
        # return  stats["total_pages"] <= 100
    
    compare_root = os.path.join('..', 'doc_data', 'raw_documents')
    loader = RawFileLoader(env_flist_path=None,#'compliance_question_file_list', 
                           walk_root=os.path.join('..', 'doc_data', 'raw_documents', 'Samples - 9 Oct 2025'),
                           compare_root = compare_root,
                           include = ['files'],
                           filtering_callbacks = [filter_callback],
                        #    file_walker_callback = find_folders_two_levels_from_leaves_mem_optimized
                           )
    
    from semantic_document_splitting_layerwise_edits import parse_doc, semantic_tree_to_kge_payload, kge_payload_to_semantic_tree,build_index_terms_for_semantic_node,all_child_from_root
    from src.ocr import regen_doc_group
    from joblib import Memory
    memory = Memory(location = '.joblib')
    for f in loader:
        splitted_folder = (pathlib.Path(os.getcwd()).parent/"doc_data"/ "split_pages").absolute()
        doc_group = regen_doc_group
        parse_doc(file_loader = loader, outfolder_path = splitted_folder, exists_ok='skip')
    
def test_semantic_document_splitting_doc_group(gemini_key):
    from src.utils.file_loaders import RawFileLoader#, find_folders_two_levels_from_leaves_mem_optimized
    import os
    from functools import lru_cache
    
    file_index_root = (pathlib.Path(os.getcwd()).parent/"doc_data"/ "file_version_chains").absolute()
    @lru_cache(maxsize = 10)
    def get_chain_index(folder_path):
        from src.utils.version_chaining import VersionChainDB
        rel_path = pathlib.Path(folder_path).relative_to(compare_root)
        try:
            db = VersionChainDB(str(pathlib.Path(file_index_root) / rel_path / "file_index.sqlite"))
        except Exception as e:
            raise e
        return db
    def filter_callback (file_path):
        folder_path = pathlib.Path(file_path).parent
        db = get_chain_index(folder_path)
        # stats = db.get_canonical_page_statistics()
        # print(f"{folder_path}: {stats=}")
        # for filestat in os.listdir(doc_group_folder_path):
        tf_use_file = db.is_canonical_by_name(file_path)
        return tf_use_file
            # else:
                # print('nono')
        # return  stats["total_pages"] <= 100
    
    compare_root = os.path.join('..', 'doc_data', 'raw_documents')
    loader = RawFileLoader(env_flist_path=None,#'compliance_question_file_list', 
                           walk_root=os.path.join('..', 'doc_data', 'raw_documents', 'Samples - 9 Oct 2025'),
                           compare_root = compare_root,
                           include = ['files'],
                        #    filtering_callbacks = [filter_callback],
                        #    file_walker_callback = find_folders_two_levels_from_leaves_mem_optimized
                           )
    
    from semantic_document_splitting_layerwise_edits import parse_doc, semantic_tree_to_kge_payload, kge_payload_to_semantic_tree,build_index_terms_for_semantic_node,all_child_from_root
    from src.ocr import regen_doc_group
    from joblib import Memory
    memory = Memory(location = '.joblib')
    for f in loader:
        f_name = pathlib.Path(f).name
        doc_group = regen_doc_group(os.path.join(loader.compare_root, f), use_raw = True)
        
        document_tree, source_map = parse_doc(doc)
        cached_semantic_tree_to_kge_payload = memory.cache(semantic_tree_to_kge_payload)
        graph_to_persist = cached_semantic_tree_to_kge_payload(document_tree)
        reconstrcted_root = kge_payload_to_semantic_tree(graph_to_persist)
        assert reconstrcted_root.model_dump() == document_tree.model_dump()
        import requests
        res = requests.post("http://127.0.0.1:28110/api/document.validate_graph", json = graph_to_persist)
        res.raise_for_status()
        nodes = all_child_from_root(reconstrcted_root)
        batch_index_list = build_index_terms_for_semantic_node(nodes)        
        payload = {"index": [i.model_dump(mode='json') for i in batch_index_list]}
        for k in payload['index']:
            k.update({'doc_id': str(reconstrcted_root.node_id)})
        @memory.cache
        def get_index_entries(payload):
            res = requests.post("http://127.0.0.1:28110/api/add_index_entries", json = payload)
            return res
        res = get_index_entries(payload)
        res.raise_for_status()
        @memory.cache
        def get_upsert_result(graph_to_persist):
            res = requests.post("http://127.0.0.1:28110/api/document.upsert_tree", json = graph_to_persist)
            return res
        res = get_upsert_result(graph_to_persist)
        res.raise_for_status()
        # search using index
        res = requests.get("http://127.0.0.1:28110/api/search_index_hybrid", 
                            params = {'q' : '"6.7"'})
        res.raise_for_status()
        res = requests.get("http://127.0.0.1:28110/api/search_index_hybrid", 
                            params = {'q' : '"6.7"', "resolve_node": True})
        res.raise_for_status()
        return
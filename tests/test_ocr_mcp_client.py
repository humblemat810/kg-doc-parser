
import sys
import os
if True:    
    sys.path.append(os.path.join(".", "src"))
import pytest
import logging

import pathlib
if True:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

from utils.log import SQLiteHandler
if True:
    sqlite_handler = SQLiteHandler(os.path.join('.','logs', 'application_logs.db'))
    sqlite_handler.setLevel(logging.DEBUG)
    logger.addHandler(sqlite_handler)
    logger.debug("test's library loading")


from typing import Optional
from joblib.memory import Memory
@pytest.mark.asyncio
async def test_regen_doc_group_and_send_to_engine_for_storing():
    from pydantic import BaseModel, Field
    from typing import Dict, Any
    class TestDocumentDTO(BaseModel):
        id: Optional[str] = Field(..., description="Unique document identifier")
        content: str = Field(..., description="Text content of the document")
        type: str = Field(..., description="Type of document, e.g., 'ocr', 'pdf'")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the document")

    from ocr import regen_doc_group
    graph_rag_port = 28110
    folder_path = os.path.join("..","doc_data","split_pages","Document Samples - cleaned","Samples for sending","Document 2 - Ficticious")
    
    location = os.path.join(".cache", "test", pathlib.Path(__file__).parts[-1], "test_regen_doc_group_and_send_to_engine_for_indexing", "test_regen_doc_group")
    os.makedirs(exist_ok = True, name = location)
    memory = Memory(location = location)
    @memory.cache
    def test_regen_doc_group(folder_path):
        return regen_doc_group(folder_path = folder_path)
    doc_group = test_regen_doc_group(folder_path = folder_path)
    doc_or_doc_group = doc_group.model_dump()
    for fname, pages in doc_or_doc_group['documents'].items():
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        async with streamable_http_client(f"http://127.0.0.1:{graph_rag_port}/mcp" #,sse_read_timeout = None, timeout = None,
                                         ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()  # SDK negotiates protocol
                pages['file_name'] = fname # flatten page with file name
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {'kg_extract', 'doc_parse', "store_document"} <= names
                import json
                res = await session.call_tool("store_document",
                            arguments={"inp":TestDocumentDTO(id = fname, content=json.dumps(pages), type="ocr", metadata = None).model_dump()})
                assert res
                

@pytest.mark.asyncio
async def test_regen_doc_group_and_send_to_engine_for_kg_extract():
    from pydantic import BaseModel, Field
    from typing import Dict, Any
    class TestDocumentDTO(BaseModel):
        id: Optional[str] = Field(..., description="Unique document identifier")
        content: str = Field(..., description="Text content of the document")
        type: str = Field(..., description="Type of document, e.g., 'ocr', 'pdf'")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the document")

    from ocr import regen_doc_group
    graph_rag_port = 28110
    folder_path = os.path.join("..","doc_data","split_pages","Document Samples - cleaned","Samples for sending","Document 2 - Ficticious")
    
    location = os.path.join(".cache", "test", pathlib.Path(__file__).parts[-1], "test_regen_doc_group_and_send_to_engine_for_indexing", "test_regen_doc_group")
    os.makedirs(exist_ok = True, name = location)
    memory = Memory(location = location)
    @memory.cache
    def test_regen_doc_group(folder_path):
        return regen_doc_group(folder_path = folder_path)
    doc_group = test_regen_doc_group(folder_path = folder_path)
    doc_or_doc_group = doc_group.model_dump()
    for fname, pages in doc_or_doc_group['documents'].items():
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        async with streamable_http_client(f"http://127.0.0.1:{graph_rag_port}/mcp"# ,sse_read_timeout = None, timeout = None,
                                         ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()  # SDK negotiates protocol

                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {'kg_extract', 'doc_parse'} <= names
                import json
                
                res = await session.call_tool("kg_extract", 
                            arguments={"inp": TestDocumentDTO(id = fname, content=json.dumps(pages), type="ocr", metadata = None).model_dump()})
                assert (res.content[0].type == "json") or (res.content[0].type == "text" and json.loads(res.content[0].text))

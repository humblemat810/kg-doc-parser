import sys
import os

import logging
import pathlib
if True:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

from kg_doc_parser.utils.log import SQLiteHandler
if True:
    sqlite_handler = SQLiteHandler(os.path.join('.','logs', 'application_logs.db'))
    sqlite_handler.setLevel(logging.DEBUG)
    logger.addHandler(sqlite_handler)
    logger.debug("test's library loading")

from typing import Optional, cast

def test_batch_ocr_one_by_one(gemini_key):
    from kg_doc_parser.utils.bounded_threadpool_executor import BoundedExecutor
    bounded_executor = None# BoundedExecutor(max_workers= 6, max_pending= 100)
    try:
        from kg_doc_parser.ocr import batch_gemini_ocr_image
        batch_gemini_ocr_image(gemini_key, bounded_executor=bounded_executor)
    except Exception as e:
        logger.error(str(e))
        import traceback
        tb_str = traceback.format_exc()
        logger.error(tb_str)
        raise
    pass
def test_batch_ocr_one_by_one_tree(gemini_key):
    from kg_doc_parser.utils.file_loaders import filter_folder
    from kg_doc_parser.utils.bounded_threadpool_executor import BoundedExecutor
    bounded_executor = None# BoundedExecutor(max_workers= 3, max_pending= 12)
    from kg_doc_parser.utils.file_loaders import RawFileLoader
    def _temp_check(x: str):
        tf = x.endswith('.png') and not os.path.exists(str(x)[:-len(pathlib.Path(x).suffix)] + '.json')
        return tf
    def filter_callback(file_path):
        import pathlib
        # use_folder = pathlib.Path(file_path).parts[-3].startswith("VIC") or pathlib.Path(file_path).parts[-3].startswith("QLD")
        use_folder = int(pathlib.Path(file_path).parts[-3].rsplit('.')[-1]) <= 4 # have a grouper folder such as company name
        return use_folder
    loader = RawFileLoader(env_flist_path=None, #'ocr_file_list', 
                            walk_root=os.path.join('..', 'doc_data', 'split_pages', 'jds'),
                            compare_root = os.path.join('..', 'doc_data', 'split_pages'),
                            include = ['files'], allow_startwith_relative_paths=True,
                            filtering_callbacks = [
                                _temp_check,
                                # filter_callback,
                                #lambda x : (not os.path.exists(str(x)[:-3] + 'json'))
                            ]
                            )
    if loader:
        allowed_relative_paths = None # delegated to loader
        
    # allowed_relative_paths = list(loader)
    else:
        allowed_relative_paths = []
        if flist:=os.environ.get('ocr_file_list'):
            if os.path.exists(flist):
                if flist.endswith('.csv'):
                    with open(flist, 'r') as f:
                        for ln in f.readlines():
                            allowed_relative_paths.append(os.path.join(*(i.strip() for i in ln.split(','))))
                elif flist.endswith('.xls') or flist.endswith('.xlsx'):
                    from pandas import read_excel
                    df = read_excel(flist)
                    allowed_relative_paths = []
                    for i, row in df.iterrows():
                        new_path = os.path.join(*(row[:3]))
                        allowed_relative_paths.append(new_path)
        allowed_file_list = filter_folder()
        allowed_relative_paths += allowed_file_list
    try:
        from kg_doc_parser.ocr import batch_gemini_ocr_image
        batch_gemini_ocr_image(gemini_key, 
                               folder=os.path.join("..", 'doc_data','split_pages'), 
                               bounded_executor=bounded_executor,
                               allowed_relative_paths = allowed_relative_paths,
                               loader = loader,
                               exist_behavior = 'skip' #'rerun'
                               )
    except Exception as e:
        logger.error(str(e))
        import traceback
        tb_str = traceback.format_exc()
        logger.error(tb_str)
        raise
    pass
def test_batch_ocr(gemini_key):
    from kg_doc_parser.utils.bounded_threadpool_executor import BoundedExecutor
    bounded_executor = BoundedExecutor(max_workers= 3, max_pending= 100)
    from kg_doc_parser.utils.file_loaders import RawFileLoader
    def _temp_check(x: str):
        tf = x.endswith('.png') # and not os.path.exists(str(x)[:-len(pathlib.Path(x).suffix)] + '.json')
        return tf
    def filter_callback(file_path):
        import pathlib
        use_folder = pathlib.Path(file_path).parts[-3].startswith("VIC") or pathlib.Path(file_path).parts[-3].startswith("QLD")
        return use_folder
    loader = RawFileLoader(env_flist_path=None, #'ocr_file_list', 
                            walk_root=os.path.join('..', 'doc_data', 'split_pages', 'updated assets-20251024T020813Z-1-001'),
                            compare_root = os.path.join('..', 'doc_data', 'split_pages'),
                            include = ['files'], allow_startwith_relative_paths=True,
                            filtering_callbacks = [
                                _temp_check,
                                # filter_callback,
                                #lambda x : (not os.path.exists(str(x)[:-3] + 'json'))
                            ]
                            )
    list(loader)
    allowed_relative_paths = None
    try:
        from ocr import batch_gemini_ocr_image
        batch_gemini_ocr_image(gemini_key, 
                               folder=os.path.join("..", 'doc_data','split_pages'), 
                               bounded_executor=bounded_executor,
                               allowed_relative_paths = allowed_relative_paths,
                               loader = loader,
                                exist_behavior = 'skip'
                               )
    except Exception as e:
        logger.error(str(e))
        import traceback
        tb_str = traceback.format_exc()
        logger.error(tb_str)
        raise
    pass

def test_gemini_ocr_pages(gemini_key):
    
    import base64
    from langchain_core.messages import HumanMessage, SystemMessage
    # Replace 'image.png' with the path to your image file.
    page_file_name = "page_1.png"
    file_name = "TIA-00-ME-ORD01-2021.PDF"
    image_file_path = os.path.join("split_pages", file_name, page_file_name)

    # Open the image in binary mode and read its content.
    with open(image_file_path, "rb") as image_file:
        image_bytes = image_file.read()

    # Base64-encode the binary data.
    encoded_bytes = base64.b64encode(image_bytes)

    # Convert the encoded bytes to a UTF-8 string (optional, if you need a string representation)
    encoded_str = encoded_bytes.decode('utf-8')

    # Print the Base64-encoded string.
    #print(encoded_str)
    sys_message = SystemMessage("You are a helpful document AI that does OCR and answer simple questions"
                                "You must include spatial arrangement in the responded text. "
                                )
    img_message = HumanMessage(
    content=[
        {"type": "text", "text": "find all text in the attached png file"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded_str}"},
        },
        ],
    )
    messages = [sys_message, img_message]
    from pydantic import BaseModel, Field
    class OCRResponse(BaseModel):
        """OCR results"""
        OCR_text: str = Field(description="the OCR text results.")
        page_number: Optional[str] = Field(description='the page number identified, can be in form of roman numerals such as "i", "ii", Arabic numeral such as 1, 2, 3')
    class TextCluster(BaseModel):
        """a text cluster along with spatial information"""
        text: str = Field(description='the text content of the text cluster')
        bb_x: float  = Field(description='the centre x pixel coordinate of the text_cluster')
        bb_y: float  = Field(description='the centre x pixel coordinate of the text_cluster')
        bb_w: float  = Field(description='the bounding box width in pixel unit of the text_cluster')
        bb_h: float  = Field(description='the bounding box height in pixel unit of the text_cluster')
        cluster_number: int = Field(description="unique number of the cluster, starting from 0")
    class OCRClusterResponse(BaseModel):
        OCR_text_clusters: list[TextCluster] = Field(description="the OCR text results.")
        meaningful_ordering : list[int] = Field(description="The correct meaningful ordering of the identified text clusters")
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    assert gemini_key.startswith("AIza") # gcp keys
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    response: OCRClusterResponse | None = cast(OCRClusterResponse | None , llm.with_structured_output(OCRClusterResponse).invoke(messages))
    if response is None:
        raise Exception("LLM give None")
    response_dict = response.model_dump()
    response_dict['pdf_page_num'] = page_file_name.rsplit('.',1)[0].rsplit("_",1)[-1]
    import json
    with open(os.path.join("split_pages",file_name, page_file_name.rsplit('.',1)[0] + '.json'), 'w') as f:
        json.dump(response_dict, f)
    print(response_dict)

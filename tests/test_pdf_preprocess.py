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

    
def test_batch_split_pdf():
    from kg_doc_parser.pdf2png import batch_split_pdf
    
    #test_doc_dir = pathlib.Path(__file__).parent/"test_documents"
    # splitted_folder = (pathlib.Path('.')/ "split_pages").absolute()
    test_doc_dir=pathlib.Path(os.getcwd()).parent/"doc_data"/"raw_documents"
    splitted_folder = (pathlib.Path(os.getcwd()).parent/"doc_data"/ "split_pages").absolute()
    batch_split_pdf(test_doc_dir, splitted_folder, exists_ok='skip')
def test_batch_split_pdf_tree_with_filter():
    from kg_doc_parser.pdf2png import batch_split_pdf
    import os
    # test_doc_dir=pathlib.Path(os.getcwd()).parent/"doc_data"/"raw_documents"
    # allowed_relative_paths = []
    from kg_doc_parser.utils.file_loaders import RawFileLoader
    loader = RawFileLoader(env_flist_path=None, #'split_raw_file_list', 
                           walk_root=os.path.join('..', 'doc_data', 'raw_documents', 'updated assets-20251024T020813Z-1-001'),
                           compare_root = os.path.join('..', 'doc_data', 'raw_documents')
                           )
    # if flist := os.environ.get("split_raw_file_list"): 
    #     with open(flist, 'r') as f:
    #         for ln in f.readlines():
    #             allowed_relative_paths.append(os.path.join(*(i.strip() for i in ln.split(','))))
    splitted_folder = (pathlib.Path(os.getcwd()).parent/"doc_data"/ "split_pages").absolute()
    batch_split_pdf(file_loader = loader, outfolder_path = splitted_folder, exists_ok='skip')
    # batch_split_pdf(test_doc_dir, splitted_folder, exists_ok='skip', allowed_relative_paths=allowed_relative_paths)


def test_raw_file_loader():
    
    from kg_doc_parser.utils.file_loaders import RawFileLoader
    loader = RawFileLoader(env_flist_path='split_raw_file_list', walk_root=os.path.join('..', 'doc_data', 'raw_documents', 'Active Vendors_230625'),
                           compare_root = os.path.join('..', 'doc_data', 'raw_documents')
                           )
    list(loader)
    
def test_batch_pdf_to_png():
    import os
    import pathlib
    pathlib.Path(__file__).parent
    abs_pdf_path = os.path.abspath(os.path.join('.', 'split_pages'))
    from kg_doc_parser.pdf2png import batch_pdf2png
    batch_pdf2png(abs_pdf_path, exists_ok="skip")
def test_batch_pdf_to_png_tree_with_filter():
    import os
    import pathlib
    pathlib.Path(__file__).parent
    abs_pdf_path = os.path.abspath((pathlib.Path(os.getcwd()).parent/"doc_data"/ "split_pages").absolute())

    def filter_callback(file_path):
        import pathlib
        use_folder = int(pathlib.Path(file_path).parts[-2].rsplit('.')[-1]) <= 4 #or pathlib.Path(file_path).parts[-2].startswith("QLD")
        return use_folder
    
    from kg_doc_parser.pdf2png import batch_pdf2png
    # allowed_file_list = filter_folder() # by page
    allowed_file_list = []
    from kg_doc_parser.utils.file_loaders import RawFileLoader, find_folders_two_levels_from_leaves_mem_optimized
    loader = RawFileLoader(env_flist_path=None, #'split_raw_file_list', 
                           walk_root=os.path.join('..', 'doc_data', 'split_pages', 'jds'),
                           compare_root = os.path.join('..', 'doc_data', 'split_pages'),
                           include = ['dirs'],
                        #    filtering_callbacks = [filter_callback],
                        #    file_walker_callback = find_folders_two_levels_from_leaves_mem_optimized
                           )
    # allowed_file_list.extend(allowed_relative_paths)
    list(loader)
    batch_pdf2png(abs_pdf_path, exists_ok="skip", allowed_relative_paths = allowed_file_list, loader = loader)
        
def test_pdf_to_png():
    from pdf2image import convert_from_path

    folder_path = "split_pages"
    

    from concurrent.futures import ThreadPoolExecutor

    def process_pdf_page(pdf_path, output_path):
        # Check if the output image already exists
        if os.path.exists(output_path):
            print(f"Skipped {output_path} (already exists)")
            return

        # Convert PDF page to image
        images = convert_from_path(pdf_path, dpi=300, fmt='png')
        
        # Save each page as a separate PNG file
        for i, image in enumerate(images):
            image.save(output_path, "PNG")
            print(f"Saved {output_path}")

    def main(folder_path):
        # Define the number of worker threads
        max_workers = 5
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for root, dirs, files  in os.walk(folder_path):
                for f in files:
                    in_page_fname = f
                    out_page_fname = f"{os.path.splitext(in_page_fname)[0]}.png"
                    input_pdf: pathlib.Path = pathlib.Path(root)/in_page_fname
                    output_path = os.path.join(root, out_page_fname)
                    
                    try:
                        if os.path.exists(output_path):
                            print(f"Skipped {output_path} (already exists)")
                            continue
                        else:
                            futures.append(executor.submit(process_pdf_page, in_page_fname, output_path))
                        #split_pdf(input_pdf, out_path, exists_ok = exists_ok)
                    except Exception as e:
                        logger.exception(e)
                        print(f"error for file {input_pdf}")
            for future in futures:
                future.result()
        
    main(folder_path)


    

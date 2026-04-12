if True:
    import logging
    import logging.handlers
    import os
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())

import shutil
import tempfile
from pdf2image import convert_from_path
import platform
import pathlib
from pypdf import PdfReader, PdfWriter
import threading
import pikepdf

try:
    from .utils.file_loaders import RawFileLoader
except ImportError:  # pragma: no cover - supports top-level test imports
    from src.utils.file_loaders import RawFileLoader

def batch_split_pdf(document_folder: pathlib.Path | str | None = None, outfolder_path: str | pathlib.Path = "split_pages", exists_ok = 'skip', allowed_relative_paths: list[str] | None= None,
                    file_loader : RawFileLoader | None = None):
    cnt = 0
    assert not ((document_folder is None) and (file_loader is None))
    class old_walker_inplace(RawFileLoader):
        def __init__(self, walk_root = None, compare_root = None):
            self.walk_root: str | pathlib.Path
            if walk_root is None:
                if document_folder:
                    self.walk_root = document_folder
                else:
                    raise Exception("unreachable")
            else:
                self.walk_root = walk_root
            if compare_root is None:
                self.compare_root = self.walk_root
        def __iter__(self):
            
            for root, dirs, files  in os.walk(self.walk_root):
                for f in files:
                    if dirs == []:
                        pass
                    else:
                        continue
                    input_pdf: pathlib.Path = pathlib.Path(root)/f
                    rel_path = input_pdf.relative_to(self.compare_root)
                    if allowed_relative_paths is not None:
                        if str(rel_path) in allowed_relative_paths:
                            pass
                        else:
                            continue
                    yield rel_path
    if file_loader is None: # document_folder must not be None
        if document_folder is None:
            raise Exception("unreacheable")
        else:
            file_loader = old_walker_inplace(document_folder)
    for rel_path in file_loader:
        
        out_path = pathlib.Path(outfolder_path)/ pathlib.Path(rel_path)
        os.makedirs(out_path.parent, exist_ok = True)
        input_pdf = pathlib.Path(file_loader.compare_root) / rel_path
        if str(input_pdf).lower().endswith('.pdf'):
            
            try:
                cnt += 1
                print(f"{cnt} {str(input_pdf)}")
                split_pdf(input_pdf, out_path.parent, exists_ok = exists_ok)
            except Exception as e:
                try:
                    logger.exception(e)
                    split_pdf_with_pikepdf(input_pdf, out_path.parent, exists_ok = exists_ok)
                    print(f"error for file {input_pdf}")
                except Exception as e:
                    raise Exception("exhausted all pdf splitter")
        elif str(input_pdf).endswith('.docx'):
            continue
            from src.utils.try_docx2pdf import docx_to_paged_pdfs_with_temp
            loader = file_loader
            out_full_dir = pathlib.Path(outfolder_path)/rel_path
            f = rel_path
            docx_to_paged_pdfs_with_temp(os.path.join(loader.compare_root, f), str(out_full_dir))


def split_pdf_with_pikepdf(input_pdf_path, output_folder, exists_ok='skip'):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the PDF with pikepdf (automatically handles decryption if no password needed)
    try:
        pdf = pikepdf.open(input_pdf_path)  # Add `password=""` if needed explicitly
    except pikepdf.PasswordError:
        raise ValueError("This PDF requires a password and could not be opened.")

    num_pages = len(pdf.pages)
    print(f"Total pages found: {num_pages}")
    fname = pathlib.Path(input_pdf_path).parts[-1]  # No extension
    output_subfolder = os.path.join(output_folder, fname)
    os.makedirs(output_subfolder, exist_ok=True)
    existing_pages = [i for i in os.listdir(os.path.join(output_folder, fname)) if i.endswith('.pdf')]
    if num_pages == len(existing_pages):
        return 
    

    

    for i, page in enumerate(pdf.pages):
        output_pdf_path = os.path.join(output_subfolder, f"page_{i + 1}.pdf")
        if os.path.exists(output_pdf_path) and exists_ok == 'skip':
            print(f"Skipped (already exists): {output_pdf_path}")
            continue

        # Create a new PDF with just one page
        new_pdf = pikepdf.Pdf.new()
        new_pdf.pages.append(page)

        new_pdf.save(output_pdf_path)
        print(f"Created: {output_pdf_path}")
    return True
def split_pdf(input_pdf_path, output_folder, exists_ok = 'skip'):
    # Ensure output folder exists
    

    # Open the PDF file for reading
    reader = PdfReader(input_pdf_path)
    if reader.is_encrypted:
        reader.decrypt("")
    num_pages = len(reader.pages)
    print(f"Total pages found: {num_pages}")
    fname = pathlib.Path(input_pdf_path).name
    os.makedirs(os.path.join(output_folder, fname), exist_ok=True)
    existing_pages = [i for i in os.listdir(os.path.join(output_folder, fname)) if i.endswith('.pdf')]
    if num_pages == len(existing_pages):
        return # skip as all num pages exported
    # Loop through each page and write it to a new PDF file
    for i in range(num_pages):
        writer = PdfWriter()
        page = reader.pages[i]
        writer.add_page(page)
        
        output_pdf_path = os.path.join(output_folder, fname,  f"page_{i + 1}.pdf")
        os.makedirs(str(pathlib.Path(output_pdf_path).parent), exist_ok=True)
        with open(output_pdf_path, "wb") as out_file:
            writer.write(out_file)
        
        print(f"Created: {output_pdf_path}")


def batch_pdf2png(document_folder, outfolder_path = None, exists_ok = 'skip', allowed_relative_paths = None,
                  loader = None):

    """_summary_

    Args:
        document_folder (_type_): folder containing splitted document folder, each folder same name as pdf file
        outfolder_path (str, optional): folder containing the folder F that contain png files. F is the name of the unsplitted pdf Defaults output same folder as splitted pdf folder. 
        exists_ok (str, optional): _description_. Defaults to 'skip'.
    """
    if outfolder_path is None:
        outfolder_path = document_folder
    from src.utils.bounded_threadpool_executor import BoundedExecutor
    bounded_executor = BoundedExecutor(max_workers= 2, max_pending= 5) # num of pdf
    # for pdf_file in os.listdir(document_folder):
    #     pdf_file : str
    if loader:
        for rel_path in loader:
            single_pdf2png(os.path.join(loader.compare_root, rel_path), os.path.join(outfolder_path, rel_path), exists_ok = exists_ok)
    else:
        for root, dirs, files  in os.walk(document_folder):
            if allowed_relative_paths is not None :
                if dirs == []:
                    rel_path = str((pathlib.Path(root)).relative_to(document_folder))
                    if rel_path not in allowed_relative_paths:
                        continue
                else:
                    continue

            single_pdf2png(os.path.join(root), str(pathlib.Path(root)), exists_ok = exists_ok)


# OS-specific file locking
if platform.system() == "Windows":
    import msvcrt
    def lock_file(file_handle):
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)

    def unlock_file(file_handle):
        try:
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass

else:
    import fcntl
    def lock_file(file_handle):
        fcntl.flock(file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock_file(file_handle):
        try:
            fcntl.flock(file_handle, fcntl.LOCK_UN)
        except Exception:
            pass


def get_thread_safe_tempfile(suffix=".pdf"):
    pid = os.getpid()
    thread_name = threading.current_thread().name.replace(" ", "_")
    prefix = f"{thread_name}_{pid}_"

    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    return fd, path
def process_pdf_page(pdf_path, output_path):
    if os.path.exists(output_path):
        print(f"Skipped {output_path} (already exists)")
        return

    tmp_fd = None
    tmp_path = None
    file_handle = None

    try:
        # Create temporary file path
        tmp_fd, tmp_path = get_thread_safe_tempfile(suffix=".pdf")
        os.close(tmp_fd)  # Close the low-level fd to avoid conflicts

        # Copy the PDF to the temporary file
        shutil.copy2(pdf_path, tmp_path)

        # Open the temp file to lock
        file_handle = open(tmp_path, 'rb')
        lock_file(file_handle)

        # Convert using the temp file
        images = convert_from_path(tmp_path, dpi=300, fmt='png')

        # Save each page as image
        for i, image in enumerate(images):
            image.save(output_path, "PNG")
            print(f"Saved {output_path}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

    finally:
        # Always release lock and delete temp file
        if file_handle:
            unlock_file(file_handle)
            file_handle.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def single_pdf2png(fname, folder_path, exists_ok = 'skip'):
    """_summary_

    Args:
        fname (_type_): the raw pdf file name before splitting
        folder_path (_type_): the folder path containing the splitted pdf pages
        exists_ok (str, optional): _description_. Defaults to 'skip'.

    Raises:
        PermissionError: _description_
    """
    
    from concurrent.futures import ThreadPoolExecutor


    def convert(folder_path, fname):
        out_pdf_folder = fname # output folder
        page_pdf_files = [f for f in os.listdir(out_pdf_folder) if f.endswith('.pdf')]
        
        # Define the number of worker threads
        max_workers = 3

        # Use ThreadPoolExecutor to process PDF files concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for page_fname in page_pdf_files:
                pdf_path = os.path.join(folder_path, page_fname)
                output_path = os.path.join(out_pdf_folder, f"{os.path.splitext(page_fname)[0]}.png")
                if os.path.exists(output_path):
                    if exists_ok == "skip":
                        continue
                    elif exists_ok == "raise":
                        raise PermissionError(f"File {output_path} already exists.")
                    else: # exist_ok == "ok"
                        pass
                futures.append(executor.submit(process_pdf_page, pdf_path, output_path))

            # Wait for all futures to complete
            # it starts running only when start to be iterated. 
            for future in futures:
                future.result()
    convert(folder_path, fname)

import sys
import os, pathlib
sys.path.insert(0, str((pathlib.Path(__file__).parent.parent).absolute()))
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import os

#import logging.handlers
#logger.addHandler(logging.handlers.RotatingFileHandler(os.path.join('.', 'logs', __name__)))
from utils.log import SQLiteHandler
sqlite_handler = SQLiteHandler(os.path.join('.','logs', 'application_logs.db'))
sqlite_handler.setLevel(logging.DEBUG)
logger.addHandler(sqlite_handler)

from pydantic import BaseModel, Field, ValidationError, model_validator
import uuid

import datetime
import hashlib
import dotenv
from typing import Literal, Optional, List, Dict, Any
from joblib import Memory
memory = Memory(location = "./.version_chain")

dotenv.load_dotenv()

import sqlite3

# ====== optional PDF -> PNG renderers ======
# we try pdf2image first, but fall back to PyMuPDF if needed
try:
    from pdf2image import convert_from_path
    _HAS_PDF2IMAGE = True
except Exception:
    _HAS_PDF2IMAGE = False
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False


# ===========================================
# PDF → PNG → hash (no temp file, in-memory)
# ===========================================

def pdf_page_hashes_as_png(
    file_path: str,
    dpi: int = 300,
    algo: str = "sha256",
    img_format: str = "PNG",
) -> list[str]:
    """
    Render each page to PNG in-memory and hash the bytes.
    Returns list of hex digests in page order.
    Includes no temp-file writes.

    NOTE: this requires either pdf2image+poppler OR PyMuPDF.
    """
    try:
        if _HAS_PDF2IMAGE:
            print(f"using pdf2image to convert file {file_path}")
            images = convert_from_path(file_path, dpi=dpi)
            out: list[str] = []
            for i, img in enumerate(images):
                print(f'page-{i}', end = ' ')
                import io
                buf = io.BytesIO()
                img.save(buf, format=img_format)
                data = buf.getvalue()
                h = hashlib.new(algo)
                h.update(data)
                out.append(h.hexdigest())
            return out
    except Exception as e:
        
        if _HAS_PYMUPDF:
            print(f"using fitz/pymupdf to convert file {file_path}")
            import fitz
            doc = fitz.open(file_path)
            out: list[str] = []
            for i, page in enumerate(doc):
                print(f'page-{i}', end = ' ')
                # dpi → matrix
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                data = pix.tobytes("png")
                h = hashlib.new(algo)
                h.update(data)
                out.append(h.hexdigest())
            return out
        raise e

    raise RuntimeError(
        "No PDF renderer available. Install `pdf2image` (plus poppler) or `PyMuPDF`."
    )


# ============================================================
# smarter subsequence check: longer -> dict(hash -> [pages])
# ============================================================

from collections import defaultdict
from bisect import bisect_right

def build_pos_index(seq: list[str]) -> dict[str, list[int]]:
    """
    seq[i] = hash_at_page_i -> index[h] = sorted list of page numbers
    """
    idx: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(seq):
        idx[h].append(i)
    return idx

def is_ordered_subsequence(shorter: list[str], longer_idx: dict[str, list[int]]) -> bool:
    """
    Check that every hash in `shorter` can be found in `longer_idx`
    in strictly increasing page order. Gaps allowed.
    """
    prev_pos = -1
    for h in shorter:
        positions = longer_idx.get(h)
        if not positions:
            return False
        j = bisect_right(positions, prev_pos)
        if j == len(positions):
            return False
        prev_pos = positions[j]
    return True


# =========================
# SQLite-backed VersionChainDB class for persistent version chain management
# =========================

class VersionChainDB:
    """
    SQLite-backed class for managing multiple version chains of PDF files.
    Each chain is a linked list of nodes (PDF files) with metadata.
    Supports CRUD, append, prepend, and insert-between operations.
    """

    def __init__(self, db_path: str = "version_chains.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id INTEGER,
            file_path TEXT,
            file_size INTEGER,
            file_hash TEXT,
            prev_id INTEGER,
            next_id INTEGER,
            created_at TEXT,
            metadata_json TEXT,
            FOREIGN KEY(chain_id) REFERENCES chains(id),
            FOREIGN KEY(prev_id) REFERENCES nodes(id),
            FOREIGN KEY(next_id) REFERENCES nodes(id)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS duplicates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            file_hash TEXT,
            duplicate_of_file_name TEXT,
            duplicate_of_file_hash TEXT,
            chain_id INTEGER,
            node_id INTEGER,
            created_at TEXT
        );
        """)

        # NEW: per-page PNG-hashes
        # we store BOTH node_id (canonical) and file_path (so you can query by name)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS page_hashes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            page_num INTEGER NOT NULL,
            page_hash TEXT NOT NULL,
            render_dpi INTEGER NOT NULL DEFAULT 150,
            render_format TEXT NOT NULL DEFAULT 'PNG',
            render_algo TEXT NOT NULL DEFAULT 'sha256',
            UNIQUE (node_id, page_num),
            FOREIGN KEY(node_id) REFERENCES nodes(id)
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_page_hashes_node ON page_hashes(node_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_page_hashes_hash ON page_hashes(page_hash);")

        # optional: convenience view to see filename alongside page hashes
        cur.execute("""
        CREATE VIEW IF NOT EXISTS vw_page_hashes AS
        SELECT
            ph.id,
            ph.node_id,
            ph.file_path,
            ph.page_num,
            ph.page_hash,
            ph.render_dpi,
            ph.render_format,
            ph.render_algo
        FROM page_hashes ph;
        """)

        if not self.conn.in_transaction:
            self.conn.commit()
    def find_smaller_files_contained_in_this_sequence(self, node_id: int) -> list[dict]:
        """
        We (node_id) are the *bigger* document (or at least, we might be).
        For every other document:
        - take its FIRST page hash
        - if that hash appears in ANY page of *us*, try to match the whole smaller doc
        """
        big_seq = self.get_page_hashes(node_id)
        if not big_seq:
            return []

        big_len = len(big_seq)

        # 1) index our own pages: hash -> [positions]
        big_idx = build_pos_index(big_seq)

        # 2) to reduce rows, we only want smaller docs whose first-page hash is
        #    one of OUR hashes.
        big_hash_set = set(big_seq)
        placeholders = ",".join("?" * len(big_hash_set))

        cur = self.conn.cursor()

        if placeholders:
            cur.execute(
                f"""
                SELECT node_id, page_hash
                FROM page_hashes
                WHERE page_num = 0
                AND node_id <> ?
                AND page_hash IN ({placeholders})
                """,
                (node_id, *big_hash_set),
            )
        else:
            # big doc somehow has no hashes? weird, just return
            return []

        first_pages = cur.fetchall()

        # our own info
        cur.execute("SELECT file_path, file_hash, chain_id FROM nodes WHERE id = ?", (node_id,))
        this_row = cur.fetchone()
        this_file_path = this_row[0] if this_row else None
        this_file_hash = this_row[1] if this_row else None
        this_chain_id = this_row[2] if this_row else None

        results: list[dict] = []

        for small_id, small_first_hash in first_pages:
            # we already know small_first_hash is in our set,
            # but it can appear multiple times (same page repeated), so:
            positions = big_idx.get(small_first_hash, [])
            if not positions:
                continue

            # fetch full smaller sequence
            small_seq = self.get_page_hashes(small_id)
            small_len = len(small_seq)
            if not small_seq:
                continue

            matched = False
            for pos in positions:
                # can we fit the smaller starting at this pos?
                if pos + small_len > big_len:
                    continue
                if big_seq[pos:pos + small_len] == small_seq:
                    matched = True
                    break

            if matched:
                # hydrate smaller
                cur.execute("SELECT file_path, file_hash, chain_id FROM nodes WHERE id = ?", (small_id,))
                small_row = cur.fetchone()
                results.append({
                    "small_node_id": small_id,
                    "small_file_path": small_row[0] if small_row else None,
                    "small_file_hash": small_row[1] if small_row else None,
                    "small_chain_id": small_row[2] if small_row else None,
                    "big_node_id": node_id,
                    "big_file_path": this_file_path,
                    "big_file_hash": this_file_hash,
                    "big_chain_id": this_chain_id,
                })

        return results   
    def get_nodes_with_pagecount_at_most(self, n_pages: int, exclude_node_id: int) -> list[int]:
        """
        Return node_ids whose page_count <= n_pages, excluding the given node.
        Useful when *this* node is longer and we want to see if we contain shorter ones.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT ph.node_id, COUNT(*) AS c
            FROM page_hashes ph
            GROUP BY ph.node_id
            HAVING c <= ?
        """, (n_pages,))
        out: list[int] = []
        for node_id, c in cur.fetchall():
            if node_id != exclude_node_id:
                out.append(node_id)
        return out
    def get_canonical_files(self) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT n.id, n.file_path, n.file_hash, n.file_size, n.chain_id, n.created_at
            FROM nodes AS n
            WHERE n.file_path NOT IN (
                SELECT d.file_name
                FROM duplicates AS d
                WHERE d.file_name IS NOT NULL
            )
            ORDER BY n.id
        """)
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "file_path": r[1],
                "file_hash": r[2],
                "file_size": r[3],
                "chain_id": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]    
    # --------------------------
    # page-hash helpers
    # --------------------------

    def insert_page_hashes(
        self,
        node_id: int,
        file_path: str,
        page_hashes: list[str],
        dpi: int = 300,
        render_format: str = "PNG",
        algo: str = "sha256",
    ):
        """
        Store the ordered page-hash sequence for a node, replacing old ones if any.
        """
        cur = self.conn.cursor()
        cur.execute("DELETE FROM page_hashes WHERE node_id = ?", (node_id,))
        cur.executemany(
            """
            INSERT INTO page_hashes
                (node_id, file_path, page_num, page_hash, render_dpi, render_format, render_algo)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (node_id, file_path, i, h, dpi, render_format, algo)
                for i, h in enumerate(page_hashes)
            ],
        )
        if not self.conn.in_transaction:
            self.conn.commit()

    def get_page_hashes(self, node_id: int) -> list[str]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT page_hash
            FROM page_hashes
            WHERE node_id = ?
            ORDER BY page_num ASC
        """, (node_id,))
        return [r[0] for r in cur.fetchall()]

    def get_nodes_with_pagecount_at_least(self, n_pages: int, exclude_node_id: int) -> list[int]:
        """
        Return node_ids whose page_count >= n_pages, excluding the given node.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT ph.node_id, COUNT(*) AS c
            FROM page_hashes ph
            GROUP BY ph.node_id
            HAVING c >= ?
        """, (n_pages,))
        out: list[int] = []
        for node_id, c in cur.fetchall():
            if node_id != exclude_node_id:
                out.append(node_id)
        return out
    
    def find_bigger_files_containing_this_sequence(self, node_id: int) -> list[dict]:
        """
        We (node_id) are the *smaller* candidate.
        Look for ANY other file that has our FIRST page-hash somewhere (any page),
        and also has enough pages after that to hold our whole sequence.
        Then confirm with a Python slice compare.
        """
        small_seq = self.get_page_hashes(node_id)
        if not small_seq:
            return []

        first_hash = small_seq[0]
        small_len = len(small_seq)

        cur = self.conn.cursor()
        # 1) SQL prune:
        #    - find ANY page (not just page 0) whose hash == our first page
        #    - make sure from that page to the end there are >= small_len pages
        cur.execute(
            """
            WITH maxp AS (
            SELECT node_id, MAX(page_num) AS max_page
            FROM page_hashes
            GROUP BY node_id
            )
            SELECT ph.node_id, ph.page_num, maxp.max_page
            FROM page_hashes AS ph
            JOIN maxp ON ph.node_id = maxp.node_id
            WHERE ph.page_hash = ?
            AND ph.node_id <> ?
            AND (ph.page_num + ?) <= (maxp.max_page + 1)
            """,
            (first_hash, node_id, small_len),
        )
        candidates = cur.fetchall()

        # our own info
        cur.execute("SELECT file_path, file_hash, chain_id FROM nodes WHERE id = ?", (node_id,))
        this_row = cur.fetchone()
        this_file_path = this_row[0] if this_row else None
        this_file_hash = this_row[1] if this_row else None
        this_chain_id = this_row[2] if this_row else None

        results: list[dict] = []

        for cand_id, start_page, max_page in candidates:
            big_seq = self.get_page_hashes(cand_id)
            # we already know indexing won't go out of range because of SQL condition
            if big_seq[start_page:start_page + small_len] == small_seq:
                # hydrate bigger file
                cur.execute("SELECT file_path, file_hash, chain_id FROM nodes WHERE id = ?", (cand_id,))
                big_row = cur.fetchone()
                results.append({
                    "small_node_id": node_id,
                    "small_file_path": this_file_path,
                    "small_file_hash": this_file_hash,
                    "small_chain_id": this_chain_id,
                    "big_node_id": cand_id,
                    "big_file_path": big_row[0] if big_row else None,
                    "big_file_hash": big_row[1] if big_row else None,
                    "big_chain_id": big_row[2] if big_row else None,
                })

        return results

    def mark_subdocument_duplicate(self, info: dict):
        """
        info must contain:
            small_file_path, small_file_hash, big_file_path, big_file_hash,
            small_chain_id, small_node_id
        We'll just reuse your existing duplicates table.
        """
        self.insert_duplicate(
            file_name=info["small_file_path"],
            file_hash=info["small_file_hash"],
            duplicate_of_file_name=info["big_file_path"],
            duplicate_of_file_hash=info["big_file_hash"],
            chain_id=info["small_chain_id"],
            node_id=info["small_node_id"],
        )

    # --------------------------

    def create_chain(self, name: Optional[str] = None) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO chains (name) VALUES (?)", (name,))
        if not self.conn.in_transaction:
            self.conn.commit()
        return cur.lastrowid

    def delete_chain(self, chain_id: int):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM nodes WHERE chain_id = ?", (chain_id,))
        cur.execute("DELETE FROM chains WHERE id = ?", (chain_id,))
        if not self.conn.in_transaction:
            self.conn.commit()

    def add_node(self, chain_id: int, file_root: str, file_path: str, file_size: int, file_hash: str, 
                 position: str = "append", ref_node_id: Optional[int] = None,
                 metadata_json: Optional[str] = None) -> int:
        """
        ref_node_id : between and append is the node before the new addition, preprend is the node id prepended to
        """
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Only PDF files (.pdf) are allowed.")
        cur = self.conn.cursor()
        created_at = datetime.datetime.now().isoformat()
        # Find head/tail for prepend/append
        if position == "prepend":
            cur.execute("SELECT id FROM nodes WHERE chain_id = ? AND prev_id IS NULL", (chain_id,))
            head = cur.fetchone()
            prev_id = None
            next_id = head[0] if head else None
            # Update old head's prev_id
            if head:
                cur.execute("UPDATE nodes SET prev_id = NULL WHERE id = ?", (head[0],))
        elif position == "append":
            cur.execute("SELECT id FROM nodes WHERE chain_id = ? AND next_id IS NULL", (chain_id,))
            tail = cur.fetchone()
            prev_id = tail[0] if tail else None
            next_id = None
            # Update old tail's next_id
            if tail:
                cur.execute("UPDATE nodes SET next_id = NULL WHERE id = ?", (tail[0],))
        elif position == "between":
            if ref_node_id is None:
                raise ValueError("ref_node_id must be provided for 'between' insertion.")
            # Insert after ref_node_id
            cur.execute("SELECT next_id FROM nodes WHERE id = ?", (ref_node_id,))
            next_id = cur.fetchone()
            next_id = next_id[0] if next_id else None
            prev_id = ref_node_id
            # Update links
            cur.execute("UPDATE nodes SET next_id = NULL WHERE id = ?", (ref_node_id,))
            if next_id:
                cur.execute("UPDATE nodes SET prev_id = NULL WHERE id = ?", (next_id,))
        else:
            raise ValueError("position must be 'append', 'prepend', or 'between'.")
        cur.execute("""
            INSERT INTO nodes (chain_id, file_path, file_size, file_hash, prev_id, next_id, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (chain_id, file_path, file_size, file_hash, prev_id, next_id, created_at, metadata_json))
        node_id = cur.lastrowid
        # Update neighbors
        if position == "prepend" and next_id:
            cur.execute("UPDATE nodes SET prev_id = ? WHERE id = ?", (node_id, next_id))
        if position == "append" and prev_id:
            cur.execute("UPDATE nodes SET next_id = ? WHERE id = ?", (node_id, prev_id))
        if position == "between":
            cur.execute("UPDATE nodes SET next_id = ? WHERE id = ?", (node_id, prev_id))
            if next_id:
                cur.execute("UPDATE nodes SET prev_id = ? WHERE id = ?", (node_id, next_id))
        if not self.conn.in_transaction:
            self.conn.commit()

        try:
            page_hashes = pdf_page_hashes_as_png(os.path.join(file_root, file_path))
            self.insert_page_hashes(
                node_id=node_id,
                file_path=file_path,
                page_hashes=page_hashes,
                dpi=150,
                render_format="PNG",
                algo="sha256",
            )

            # 1) I am short -> check longer ones
            longer_matches = self.find_bigger_files_containing_this_sequence(node_id)
            for info in longer_matches:
                self.mark_subdocument_duplicate(info)

            # 2) I am long  -> check shorter ones
            shorter_matches = self.find_smaller_files_contained_in_this_sequence(node_id)
            for info in shorter_matches:
                self.mark_subdocument_duplicate(info)

        except Exception as e:
            logger.exception(f"Error computing/storing page hashes for {file_path}: {e}")
            self.conn.rollback()
            raise e
            
        return node_id

    def list_all_chains(self):
        return [self.get_chain(chain['id']) for chain in self.find_chains()]

    def get_chain(self, chain_id: int) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        # Find head node
        cur.execute("SELECT id FROM nodes WHERE chain_id = ? AND prev_id IS NULL", (chain_id,))
        head = cur.fetchone()
        if not head:
            return []
        node_id = head[0]
        chain = []
        while node_id:
            cur.execute("SELECT id, file_path, file_size, file_hash, prev_id, next_id, created_at, metadata_json FROM nodes WHERE id = ?", (node_id,))
            row = cur.fetchone()
            if not row:
                break
            node = {
                "id": row[0],
                "file_path": row[1],
                "file_size": row[2],
                "file_hash": row[3],
                "prev_id": row[4],
                "next_id": row[5],
                "created_at": row[6],
                "metadata_json": row[7]
            }
            chain.append(node)
            node_id = row[5]  # next_id
        return chain

    def update_node(self, node_id: int, **fields):
        cur = self.conn.cursor()
        allowed = {"file_path", "file_size", "file_hash", "metadata_json"}
        updates = []
        values = []
        for k, v in fields.items():
            if k in allowed:
                updates.append(f"{k} = ?")
                values.append(v)
        if not updates:
            return
        values.append(node_id)
        cur.execute(f"UPDATE nodes SET {', '.join(updates)} WHERE id = ?", values)
        if not self.conn.in_transaction:
            self.conn.commit()

    def delete_node(self, node_id: int):
        cur = self.conn.cursor()
        # Relink neighbors
        cur.execute("SELECT prev_id, next_id FROM nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
        if row:
            prev_id, next_id = row
            if prev_id:
                cur.execute("UPDATE nodes SET next_id = ? WHERE id = ?", (next_id, prev_id))
            if next_id:
                cur.execute("UPDATE nodes SET prev_id = ? WHERE id = ?", (prev_id, next_id))
        cur.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        # also delete page-hashes for this node
        cur.execute("DELETE FROM page_hashes WHERE node_id = ?", (node_id,))
        if not self.conn.in_transaction:
            self.conn.commit()

    def find_chains(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, name FROM chains")
        return [{"id": row[0], "name": row[1]} for row in cur.fetchall()]

    def find_nodes(self, chain_id: int) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, file_path, file_size, file_hash, prev_id, next_id, created_at, metadata_json FROM nodes WHERE chain_id = ?", (chain_id,))
        return [
            {
                "id": row[0],
                "file_path": row[1],
                "file_size": row[2],
                "file_hash": row[3],
                "prev_id": row[4],
                "next_id": row[5],
                "created_at": row[6],
                "metadata_json": row[7]
            }
            for row in cur.fetchall()
        ]
    def get_canonical_for_hash(self, file_hash: str):
        """
        Return (id, file_path, chain_id) of the earliest node we have for this hash.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, file_path, chain_id
            FROM nodes
            WHERE file_hash = ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (file_hash,),
        )
        return cur.fetchone()
    def insert_duplicate(self, file_name, file_hash, duplicate_of_file_name, duplicate_of_file_hash, chain_id=None, node_id=None):
        cur = self.conn.cursor()
        created_at = datetime.datetime.now().isoformat()
        cur.execute("""
            INSERT INTO duplicates (file_name, file_hash, duplicate_of_file_name, duplicate_of_file_hash, chain_id, node_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (file_name, file_hash, duplicate_of_file_name, duplicate_of_file_hash, chain_id, node_id, created_at))
        if not self.conn.in_transaction:
            self.conn.commit()

    def find_duplicate_by_name(self, file_name):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM duplicates WHERE file_name = ?", (file_name,))
        return cur.fetchall()

    def find_duplicate_by_hash(self, file_hash):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM duplicates WHERE file_hash = ?", (file_hash,))
        return cur.fetchall()
    def get_canonical_page_statistics(self) -> dict:
        """
        Compute statistics on canonical (non-duplicate) documents.

        Returns:
            dict with:
                - total_canonical_docs (int)
                - total_pages (int)
                - details (list of dict) → each with {file_path, page_count}
        
        Canonical = nodes whose file_path NOT in duplicates.file_name.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT
                n.file_path,
                COUNT(ph.page_num) AS page_count
            FROM nodes AS n
            JOIN page_hashes AS ph
                ON ph.node_id = n.id
            WHERE n.file_path NOT IN (
                SELECT d.file_name FROM duplicates AS d
                WHERE d.file_name IS NOT NULL
            )
            GROUP BY n.id
            ORDER BY page_count DESC;
        """)
        rows = cur.fetchall()

        stats = {
            "total_canonical_docs": len(rows),
            "total_pages": sum(r[1] for r in rows),
            "details": [{"file_path": r[0], "page_count": r[1]} for r in rows]
        }
        return stats
    def name_exists(self, file_name):
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM nodes WHERE file_path = ? LIMIT 1", (file_name,))
        if cur.fetchone():
            print('match from nodes.file_path')
            return True
        cur.execute("SELECT 1 FROM duplicates WHERE file_name = ? LIMIT 1", (file_name,))
        if cur.fetchone():
            print('match from duplicates.file_name')
            return True
        return False

    def hash_exists(self, file_hash):
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM nodes WHERE file_hash = ? LIMIT 1", (file_hash,))
        if cur.fetchone():
            return True
        cur.execute("SELECT 1 FROM duplicates WHERE file_hash = ? LIMIT 1", (file_hash,))
        if cur.fetchone():
            return True
        return False

    def is_duplicate_name_or_hash(self, file_name, file_hash):
        return self.name_exists(file_name) or self.hash_exists(file_hash)

    def close(self):
        self.conn.close()
    def is_canonical_by_name(self, file_name: str) -> bool:
        """
        Decide if this file_path is the canonical/kept one.

        Rules (based on your current usage):
        - if this file_name appears in duplicates.file_name  -> NOT canonical
        (because you always put the thing-to-hide on the left)
        - else -> canonical
        - if we also want to be careful with old rows, we can fallback to hash
        """
        cur = self.conn.cursor()

        # 1) if it's explicitly marked as a duplicate, it's not canonical
        cur.execute("SELECT 1 FROM duplicates WHERE file_name = ? LIMIT 1", (file_name,))
        if cur.fetchone():
            return False

        # 2) try to get its hash from nodes
        cur.execute("""
            SELECT file_hash
            FROM nodes
            WHERE file_path = ?
            LIMIT 1
        """, (file_name,))
        row = cur.fetchone()
        if row:
            file_hash = row[0]
        else:
            # maybe it only lives in duplicates table (rare, but let's check)
            cur.execute("""
                SELECT file_hash
                FROM duplicates
                WHERE file_name = ?
                LIMIT 1
            """, (file_name,))
            row = cur.fetchone()
            if row:
                file_hash = row[0]
            else:
                # not in nodes, not in duplicates: we don't know it -> treat as canonical/new
                return True

        # 3) (optional) if you want to be extra safe:
        #    check if there is an entry "some other file" -> this hash
        #    but because your direction is always "duplicate file_name -> canonical duplicate_of_file_name",
        #    step (1) is usually enough.
        return True
    def get_non_duplicate_files(self) -> list[dict]:
        """
        Return all nodes that are not listed as duplicates.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT n.id, n.file_path, n.file_hash, n.chain_id, n.created_at
            FROM nodes AS n
            WHERE n.file_hash NOT IN (
                SELECT d.file_hash FROM duplicates AS d
            )
            ORDER BY n.created_at ASC;
        """)
        rows = cur.fetchall()
        return [
            {"id": r[0], "file_path": r[1], "file_hash": r[2],
            "chain_id": r[3], "created_at": r[4]}
            for r in rows
        ]

# --------------------------------------------------------------------------------
# The rest is your original LLM + ingestion logic, mostly unchanged
# --------------------------------------------------------------------------------

class FileMetadata(BaseModel):
    "metadata about a file"
    document_name: str = Field(..., description="document id")
    file_size: int = Field(..., description = "file size in bytes")
    date_modified : str = Field(..., description = "date modified / copied to the analysis system")
    date_created : str = Field(..., description = "date created")
    file_hash : Optional[str]  = Field(default = None,  description = "file hash")
    def __hash__(self):
        if self.file_hash is None:
            raise ValueError("file_hash must be provided for the model to be hashable")
        return int(self.file_hash, base=16)

@memory.cache
def get_file_hash(file_path, last_modified, size_bytes, algorithm="sha256", block_size=65536, ):
    """Compute a hash for the given file using the specified algorithm."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def get_folder_metadata(folder_path, hash_algorithm="sha256"):
    metadata_list = []
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            file_path = os.path.join(root, name)
            stats = os.stat(file_path)
            file_hash = get_file_hash(file_path, last_modified = stats.st_mtime,
                                      size_bytes= stats.st_size,
                                      algorithm=hash_algorithm)
            metadata_list.append(
                FileMetadata(
                    document_name = name,
                    file_size = stats.st_size,
                    file_hash = file_hash,
                    date_modified= str(datetime.datetime.fromtimestamp(stats.st_mtime)),
                    date_created =str(datetime.datetime.fromtimestamp(stats.st_birthtime ))
                )
            )
    return metadata_list

class FileVersion(BaseModel):
    "representing a file version"
    filename: str = Field(..., description = "the filename of the current version")
    prev: Optional[str] = Field(..., description = "The file name of the previous version. If it is brandnew not superceding/ overwriting any other, set None/Null. ")
    supercede_reason : str = Field(..., description = "Why this version fully supercede the previous.")
    supercede_mode : Literal["Duplicate", "FileEdit", "ContractUpdate", "N/A"]  = Field(..., description = """
                                        the mode of superceding, Duplicate means the file is just a duplicated copy. 
                                        FileEdit is small edit that does not change any activated contract terms. FileEdit is applicable to any contract file change across drafts. 
                                        Contract update is the update that truely reflect the signed contract with intention to update. 
                                        "N/A" when there is nothing to supercede when it is the first version. """)

class VersionChain(BaseModel):
    "A linked list that the represent the evolution of a file. "
    chain: list[FileVersion] = Field(..., description = "A single file mutation chain, first element is the root, subsequent files is the mutated version of the preceding. ")
    pass

class FileVersionChainingResponse(BaseModel):
    "Answer response format of file version chaining"
    reasoning: str = Field(..., description = "reasoning at overall response level of thinking")
    chains: list[VersionChain] = Field(..., description = "a array/ list of linked list with first element the raw verion before any changes. The next element is the file version that overwrites/ supercede the previous version. ")
    root_agreement : str  = Field(..., description = "The file name of the origin/master agreement that covers everything before any term variations are applied to. ")
    pass

@memory.cache
def version_chain(metadata_list: list[FileMetadata], model = 'gemini-2.5-pro', attempt = 0):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(
        "You need to provide version chain to a list of given file data. You need to sort out which one is superceded by which. "
        "You are given a list of file metadata and output file versioning chains. You need to reason through why one is before or after another file. "
        "You need to cover all files, if it has no previous version. Give result as one or more file version chains. "
        "There are some subtle file hint they are the same. Sometimes the same file chain do not share same name but some part of the file is the same. "
        "The same file version chain may have the file renamed but preserve similar meaning, partially renamed or truncated. "
        "For example a file maybe called in its original version `schedule 1.pdf` but the next version available is not will be schedule 1v4_final_final.pdf with potentially v2 (version 2) and v3 missing and a strange final suffix appended. \n"
        "Of course there can be some simpler case maybe just as simple as `MSA Company name v1.pdf` and the next is simply MSA Company name v2.pdf"
        "Your job is to track these chains as much as possible. "
        "Example, the evolution / edit / terms changes of scheule A should be distinct from the changes of schedule B. "
        "Remember, we are dealing with version, do not add schedule 3 after schedule 2, but only add schedule 3 version 2 after schedule 3. The do not use reading order (e.g. section 4 after section 3). "
        "We are only concerned with estimating changes made to the same section. If it is a separate section/ contract. Or the varaition is dealing with different terms, they should follow its own new chain. "
        "In case the file is purely copy and pasted with a different number at the end, try to choose latest, with largest number as the newer version. "
        "The given file list comes from os files. Each file must show up exactly once in the answer chain. "
    ),
    HumanMessage(f"{[i.model_dump(exclude = ['file_hash']) for i in metadata_list]}")]

    llm = ChatGoogleGenerativeAI(model = model)
    cnt = 0
    max_cnt = 4
    while cnt <= max_cnt:
        chaining_result = llm.with_structured_output(FileVersionChainingResponse, include_raw = True).invoke(messages)
        chains = chaining_result['parsed'].model_dump()['chains']
        files = [cc['filename'] for c in chains for cc in c['chain']]
        from collections import Counter
        counter = Counter(files)
        duplicated = []
        for i in counter.most_common():
            if i[1] > 1:
                duplicated.append(i)
        if duplicated:
            messages.append(f"Trial answer : the field chains= {chains}")
            messages.append(f"files duplicated: {str(duplicated)}")
        else:
            break
        cnt += 1
        if cnt == max_cnt:
            raise(Exception(f"Max retry {max_cnt} reached" ))
    return chaining_result['parsed'].model_dump()['chains']

@memory.cache
def dedup_llm_pick_newest(meta_list_dumped, model = 'gemini-2.5-flash'):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage

    representative_file_name = None
    name_list = [i['document_name'] for  i in meta_list_dumped]

    class DedupResponse(BaseModel):
        reasoning: str = Field(..., description = "The reasoning steps to the final answer. ")
        representative_file_name: str = Field(..., description = "The file name of the most representative file. ")

    cnt = 0
    while representative_file_name not in name_list:
        messages = [SystemMessage("You are given a list of files sharing the same file hash and you need to choose one that is the most representative. "),
                    HumanMessage(f"{meta_list_dumped}")]
        llm = ChatGoogleGenerativeAI(model = model)
        res = llm.with_structured_output(DedupResponse, include_raw = True).invoke(messages)
        if not res.get('parsing_error'):
            representative_file_name = res['parsed'].representative_file_name
            if representative_file_name not in name_list:
                messages.append(SystemMessage("Error, the answer mentioned file name does not exist in any of the file at all. Only choose the exact file name in the list. "))
            else:
                return representative_file_name
        else:
            pass
        cnt +=1
        if cnt > 5:
            raise Exception("LLM error, retried max reached and cannot choose dedup filename.")
    pass

def dedup(list_meta: list[FileMetadata]):
    d: dict[str, set[FileMetadata]] = {}
    for meta in list_meta:
        m = meta.model_dump(exclude = ['file_hash'])
        if meta.file_hash not in d:
            d[meta.file_hash] = [m]
        else:
            d[meta.file_hash].append(m)
    res = {}
    dup_res = []
    for hs, meta_set in d.items():
        if len(meta_set) > 1:
            dup = {i['document_name']: i for i in meta_set}
            logger.info(f"deduping set {';'.join(list(dup))}")
            src = dedup_llm_pick_newest(meta_set)
            res[hs] = dup.pop(src)
            dup_res.extend([FileVersion.model_validate({"filename": d['document_name'], "prev": src, "supercede_reason": f"duplicate of file {src}",  "supercede_mode" : "Duplicate"})  for d in dup.values()])
        else:
            res[hs] = list(meta_set)[0]
    return res, dup_res

def check_missing_db(db: VersionChainDB, d_hash_to_meta):
    # Get all filenames in DB
    all_db_files = set()
    for chain in db.find_chains():
        for node in db.get_chain(chain['id']):
            all_db_files.add(node['file_path'])
    available_files = set(meta.document_name for meta in d_hash_to_meta.values())
    missing = available_files - all_db_files
    return missing

class FileAddition(FileVersion):
    "file addition"
    filename: str = Field(..., description = 'the file name of the new file')
    prev: Optional[str]  = Field(None, description = 'the file name its previous version, set Null or None if it belong to new chain')
    next: Optional[str]  = Field(None, description = 'the file name its next version, use only when this file is inserted to the beginning of an existing file version chain. ') 
    supercede_reason : str = Field(..., description = "Why this version fully supercede the previous.")
    supercede_mode : Literal["Duplicate", "FileEdit", "ContractUpdate", "N/A"] = Field(..., description = """
                                        the mode of superceding, Duplicate means the file is just a duplicated copy. 
                                        FileEdit is small edit that does not change any activated contract terms. FileEdit is applicable to any contract file change across drafts. 
                                        Contract update is the update that truely reflect the signed contract with intention to update. 
                                        "N/A" when there is nothing to supercede when it is the first version. """)
    @model_validator(mode = 'after')
    def only_one_prev_next_none(self):
        if (self.prev is None) or (self.next is None):
            return self
        else:
            raise Exception('at most one of  prev or next is None')

class AddFilesResponse(BaseModel):
    "the result of the file addition"
    reasoning: str  = Field(..., description = 'reasoning in top level perspective')
    additions : list[FileAddition] = Field(..., description = 'list of file additions')

@memory.cache
def add_new_file_to_existing_chains(chains, all_d_hash_to_meta: dict[str, FileMetadata], new_file_name, model = 'gemini-2.5-pro', attempt = 0 ) -> AddFilesResponse:
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [SystemMessage("Given existing file versioning chain and a new file with metadata of all files, you need to decide the position of the new file in the version chain. "
                              "All files have distinct file hash. "
    "if existing chain is file_v1, file_v1.1, file2 and a new file file_v_1.5 and you think it should be inserted between file_v1.1 and file2. "
    "Remember, we are dealing with version, do not add schedule 3 after schedule 2, but only add schedule 3 version 2 after schedule 3. The do not use reading order (e.g. section 4 after section 3). "
    "We are only concerned with estimating changes made to the same section. If it is a separate section/ contract. Or the varaition is dealing with different terms, they should follow its own new chain. "
     "then set the prev of file_v_1.5 to be file_v1.1, remember if it is newer than one but looks older than another, you need to reason about the possibility of insertion instead of appending directly at the end. "),
                HumanMessage(f"Existing version chain: {chains}" + "\n\n" +
                             f"Metadata file-hash to file-meta lookup for all files: {[i.model_dump(exclude = set(['file_hash'])) for i in all_d_hash_to_meta.values()]}" + "\n\n" +
                             f"New file to add: {new_file_name}"
                             ),
                
                ]
    from langchain_google_genai import ChatGoogleGenerativeAI
    max_retry = 3
    n = 0
    while True:
        llm = ChatGoogleGenerativeAI(model = model)
        res = llm.with_structured_output(AddFilesResponse, include_raw = True).invoke(messages)
        if res.get("parsing_error"):
            n += 1
            if n >= max_retry:
                raise Exception(f"Max Retry reached {max_retry}")
        else:
            return res.get('parsed')

def apply_chain_updates_db(db: VersionChainDB, updates, all_meta: dict[str, FileMetadata]):
    additions: list[FileAddition] = updates.additions
    changes_made = {"added_nodes": [], "added_chains": []}
    for addition in additions:
        meta = all_meta.get(addition.filename)
        if addition.prev is None and addition.next is None:
            # New chain
            chain_id = db.create_chain(name=f"Chain_for_{addition.filename}")
            changes_made["added_chains"].append(chain_id)
            changes_made["added_nodes"].append(db.add_node(
                chain_id=chain_id,
                file_path=addition.filename,
                file_size=meta.file_size if meta else 0,
                file_hash=(meta.file_hash if (meta and meta.file_hash is not None) else ""),
                position="append",
                metadata_json=str(addition.model_dump())
            ))
        else:
            # Find the chain and reference node for insertion
            target_chain_id = None
            ref_node_id = None
            position = None
            for chain in db.find_chains():
                nodes = db.get_chain(chain['id'])
                # Insert at beginning (prepend)
                if addition.prev is None and addition.next is not None:
                    for node in nodes:
                        if node['file_path'] == addition.next:
                            target_chain_id = chain['id']
                            ref_node_id = node['id']
                            position = "prepend"
                            if node['prev_id']:  # next existed, so it's actually "between"
                                position = "between"
                            else:
                                position = "prepend"
                            break
                # Insert at end (append)
                elif addition.prev is not None and addition.next is None:
                    for node in nodes:
                        if node['file_path'] == addition.prev:
                            target_chain_id = chain['id']
                            ref_node_id = node['id']
                            if node['next_id']:
                                position = "between"
                            else:
                                position = "append"
                            break
                # Insert between
                elif addition.prev is not None and addition.next is not None:
                    for node in nodes:
                        if node['file_path'] == addition.prev:
                            target_chain_id = chain['id']
                            ref_node_id = node['id']
                            position = "between"
                            break
                if target_chain_id is not None:
                    break
            if target_chain_id is not None and ref_node_id is not None and position is not None:
                changes_made["added_nodes"].append(db.add_node(
                    chain_id=target_chain_id,
                    file_path=addition.filename,
                    file_size=meta.file_size if meta else 0,
                    file_hash=(meta.file_hash if (meta and meta.file_hash is not None) else ""),
                    position=position,
                    ref_node_id=ref_node_id,
                    metadata_json=str(addition.model_dump())
                ))
            # if position is None, skip for now
    return changes_made


# ---------------------------
# optional: backfill
# ---------------------------
def backfill_page_hashes(db: VersionChainDB, file_root, dpi: int = 300):
    """
    Run once on an existing DB to populate page_hashes and page-based duplicates.
    """
    cur = db.conn.cursor()
    cur.execute("SELECT id, file_path FROM nodes")
    rows = cur.fetchall()
    for node_id, file_path in rows:
        full_path = os.path.join(file_root, file_path)
        if not file_path.lower().endswith(".pdf"):
            continue
        if not os.path.exists(full_path):
            continue
        try:
            ph = pdf_page_hashes_as_png(full_path, dpi=dpi)
            db.insert_page_hashes(node_id, file_path, ph, dpi=dpi)
            contained_in = db.find_bigger_files_containing_this_sequence(node_id)
            for info in contained_in:
                db.mark_subdocument_duplicate(info)
        except Exception as e:
            logger.exception(f"Backfill page-hash failed for {file_path}: {e}")
            raise e


if __name__ == "__main__":
    from pdf2png import RawFileLoader
    import pandas as pd
    in_compare_root = os.path.join('..', 'doc_data', 'raw_documents')
    loader = RawFileLoader(
        env_flist_path=None,
        walk_root=os.path.join('..', 'doc_data', 'raw_documents', 'Samples - 9 Oct 2025'),
        compare_root=in_compare_root,
        include=['dirs']
    )
    filechain_folder_root = os.path.join('..', 'doc_data', 'file_version_chains')

    for rel_in_path in loader:
        foldername = os.path.join(in_compare_root, rel_in_path)
        out_folder = os.path.join(filechain_folder_root, rel_in_path)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        folder_meta = get_folder_metadata(foldername)
        print(folder_meta)

        db = VersionChainDB(db_path=os.path.join(out_folder, "file_index.sqlite"))

        # 1) first-level name/hash dedup against existing DB rows
        filtered_meta = []
        for m in folder_meta:
            if db.name_exists(m.document_name):
                # If name exists, skip entirely
                continue
            elif db.hash_exists(m.file_hash):
                # If name does NOT exist, but hash exists, insert duplicate
                canon = db.get_canonical_for_hash(m.file_hash)
                if canon:
                    canon_id, canon_path, canon_chain_id = canon
                    db.insert_duplicate(
                        file_name=m.document_name,               # this new one is the duplicate
                        file_hash=m.file_hash,
                        duplicate_of_file_name=canon_path,       # point to the canonical existing one
                        duplicate_of_file_hash=m.file_hash,
                        chain_id=canon_chain_id,
                        node_id=None,                            # we don't have a node_id for the new one yet
                    )
                db.conn.commit()
                continue
            else:
                # If neither exists, process as new file
                filtered_meta.append(m)

        # 2) LLM-driven dedup for the new batch
        d_hash_to_meta, dup = dedup(filtered_meta)
        df_dup = pd.DataFrame([d.model_dump(exclude=['file_hash']) for d in dup])
        d_hash_to_meta = {k: FileMetadata.model_validate(v) for k, v in d_hash_to_meta.items()}
        df_to_concat = [df_dup]
        attempt = 0
        chains_response = version_chain(list(d_hash_to_meta.values()), attempt=attempt)
        chains = chains_response
        if attempt == 0:
            # emulate hallucination case when a file meta missing
            popped_chain = chains.pop(-1)
            for chain in chains:
                if len(chain['chain']) > 3:
                    popped_doc = chain['chain'].pop(1)
                    break

        filename_to_meta = {i.document_name: i for i in filtered_meta}
        try:
            db.conn.execute("BEGIN")

            # persist each duplicate
            for file_version in dup:
                dup_meta = next((m for m in filtered_meta if m.document_name == file_version.filename), None)
                rep_meta = next((m for m in filtered_meta if m.document_name == file_version.prev), None)
                db.insert_duplicate(
                    file_name=file_version.filename,
                    file_hash=dup_meta.file_hash if dup_meta else None,
                    duplicate_of_file_name=file_version.prev,
                    duplicate_of_file_hash=rep_meta.file_hash if rep_meta else None,
                    chain_id=None,
                    node_id=None
                )

            # persist each chain
            for chain_dict in chains_response:
                chain_id = db.create_chain(name=str(uuid.uuid1()))
                for file_version in chain_dict['chain']:
                    meta = filename_to_meta.get(file_version['filename'])
                    db.add_node(
                        chain_id=chain_id,
                        file_path=file_version['filename'],
                        file_size=meta.file_size if meta else 0,
                        file_hash=(meta.file_hash if (meta and meta.file_hash is not None) else ""),
                        position="append",
                        metadata_json=str(file_version)
                    )

            db.conn.commit()
        except Exception as e:
            db.conn.rollback()
            print(f"Error inserting chain: {e}")
            raise e

        df = pd.DataFrame([j for i in chains for j in i['chain']])
        df_to_concat.append(df)
        missing = check_missing_db(db, d_hash_to_meta)
        max_retry = 6
        while missing:
            updates = add_new_file_to_existing_chains(chains, d_hash_to_meta, missing, attempt=attempt)
            apply_chain_updates_db(db, updates, filename_to_meta)
            missing = check_missing_db(db, d_hash_to_meta)
            attempt += 1
            if attempt > max_retry:
                raise Exception(f"Max retry {max_retry} reached without fixing version chain")
        import pathlib
        df_out = pd.concat(df_to_concat)
        df_out.to_csv(pathlib.Path(foldername).parts[-1] + '.csv')

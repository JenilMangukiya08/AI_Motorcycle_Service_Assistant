import pdfplumber
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
import fitz  
from PIL import Image
import pytesseract
import os
import pickle
import faiss
import numpy as np
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
#---------------------------------------------------------
#PDF FILES IMPORT
#---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANUALS_DIR = os.path.join(BASE_DIR, "manuals")
PROJECT_ROOT = os.path.dirname(BASE_DIR)

def get_all_pdfs():

    all_pdfs = []

    for bike_name in os.listdir(MANUALS_DIR):

        bike_folder = os.path.join(
            MANUALS_DIR,
            bike_name
        )

        if not os.path.isdir(bike_folder):
            continue

        for file in os.listdir(bike_folder):

            if file.lower().endswith(".pdf"):

                all_pdfs.append({
                    "bike": bike_name,
                    "pdf_path": os.path.join(
                        bike_folder,
                        file
                    )
                })

    return all_pdfs

text_content=[]
table_content=[]
image_data=[]


#---------------------------------------------------------
#DATA CLEANING
#---------------------------------------------------------
def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'(?m)^\d+\s*$', '', text)
    text = re.sub(r'(?m)^NOTES\s*:-\s*$', '', text)
    text = re.sub(r'(?m)^1\.3\s+VEHICLE\s+SPECIFICATIONS\s*$','',text,flags=re.IGNORECASE)

    return text.strip()

def normalize(text):
    return re.sub(r'\s+', ' ', text or "").strip().lower()

#---------------------------------------------------------
#SPELLING MISSTAKE CHECKER
#---------------------------------------------------------
spell = SpellChecker()

def correct_text(text):
    words = text.split()

    corrected = []

    for word in words:

        # keep numbers unchanged
        if word.isdigit():
            corrected.append(word)
            continue

        correction = spell.correction(word)

        corrected.append(correction if correction else word)

    return " ".join(corrected)

#---------------------------------------------------------
#IMAGE FROM TABLE
#---------------------------------------------------------
DPI   = 200
SCALE = DPI / 72
def find_desc_photo_indices(tbl_data):
        for ri, row in enumerate(tbl_data):
            if not row:
                continue
            cells = [normalize(c) for c in row]
            desc_idx  = next((i for i, c in enumerate(cells) if "desc"  in c), None)
            photo_idx = next((i for i, c in enumerate(cells) if "photo" in c), None)
            if desc_idx is not None and photo_idx is not None:
                return desc_idx, photo_idx, ri
        # Fallback: if first cell is a number assume standard layout
        if tbl_data and tbl_data[0] and len(tbl_data[0]) >= 3:
            if re.match(r'^\d+$', normalize(tbl_data[0][0])):
                return 1, 2, -1
        return None, None, None

#---------------------------------------------------------
#FLOATING IMAGES
#---------------------------------------------------------
def get_section(blocks):
    for b in blocks:
        t = re.sub(r'\s+', ' ', b[4]).strip()
        if re.match(r'^\d+\.\d+', t) and 5 < len(t) < 120:
            return t
    return ""

def get_best_label(blocks, img_y0, img_y1):
    heading_pat = re.compile(r'^[ivxlcIVXLC]+[.)]\s+|^\d+\.\d+|^[A-Z][A-Z\s\-/]{3,}$')
    skip_pat    = re.compile(r'^(\d{1,3}|NOTES\s*:-?|!\s*CAUTION|!\s*WARNING|NOTE)$', re.I)

    # Part 1: nearest heading whose TOP is above image top
    best_heading, best_h_dist = "", float("inf")
    for b in blocks:
        if b[2] > 360: continue            
        if b[1] >= img_y0: continue       
        t = re.sub(r'\s+', ' ', b[4]).strip()
        if not t or len(t) < 4: continue
        if not heading_pat.match(t): continue
        dist = img_y0 - b[1]
        if dist < best_h_dist:
            best_h_dist, best_heading = dist, t

    img_mid = (img_y0 + img_y1) / 2
    best_near, best_n_dist = "", float("inf")
    for b in blocks:
        if b[2] > 360: continue
        t = re.sub(r'\s+', ' ', b[4]).strip()
        if not t or len(t) < 8: continue
        if skip_pat.match(t): continue
        if heading_pat.match(t): continue  # heading already captured above
        dist = abs((b[1] + b[3]) / 2 - img_mid)
        if dist < best_n_dist:
            best_n_dist, best_near = dist, t

    if best_heading and best_near:
        return f"{best_heading} | {best_near}"[:300]
    return (best_heading or best_near or f"page {int(img_y0)} image")[:300]



#---------------------------------------------------------
#METADA FILTERING
#---------------------------------------------------------

def filter_chunks(metadata_list,
                  bike=None,
                  pdf=None,
                  page=None):

    valid_indices = []

    for idx, meta in enumerate(metadata_list):

        if bike and meta["bike"] != bike:
            continue

        if pdf and meta["pdf"] != pdf:
            continue

        if page and meta["page"] != page:
            continue

        valid_indices.append(idx)

    return valid_indices



def built_database():
    #---------------------------------------------------------
    #TEXT + TABLE + TOOL-IMAGE + INSTRUCTION-IMAGE EXTRACTION
    #(merged into a single per-PDF, single per-page pass so each
    # page is opened/rendered exactly once instead of three times)
    #---------------------------------------------------------

    os.makedirs(os.path.join(BASE_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "instr_images"), exist_ok=True)

    for pdf_info in get_all_pdfs():

        bike_name = pdf_info["bike"]
        pdf_file  = pdf_info["pdf_path"]
        pdf_name  = os.path.basename(pdf_file)
        pdf_short = os.path.splitext(pdf_name)[0]

        fitz_doc  = fitz.open(pdf_file)
        saved_pos = set()  # dedup for instruction images, per PDF

        with pdfplumber.open(pdf_file) as plumber_pdf:

            for page_index, plumber_page in enumerate(plumber_pdf.pages):
                page_num = page_index + 1

                # ---- TEXT EXTRACTION ----
                raw_text = plumber_page.extract_text()
                if raw_text:
                    cleaned = clean_text(raw_text)
                    if len(cleaned) > 20:
                        text_content.append({
                            "bike": bike_name,
                            "pdf": pdf_name,
                            "page": page_num,
                            "text": cleaned
                        })

                # ---- TABLE -> TEXT ROWS ----
                found_tables = plumber_page.find_tables()
                raw_tables   = plumber_page.extract_tables()

                if raw_tables:
                    for table in raw_tables:
                        headers = table[0]
                        for row in table[1:]:
                            if not row:
                                continue
                            row_data = []
                            for h, v in zip(headers, row):
                                h = clean_text(h) if h else ""
                                v = clean_text(v) if v else ""
                                if h and v:
                                    row_data.append(f"{h}: {v}")
                            if row_data:
                                table_content.append({
                                    "bike": bike_name,
                                    "pdf": pdf_name,
                                    "page": page_num,
                                    "text": "[TABLE ROW]\n" + " | ".join(row_data)
                                })

                # ---- Decide if this page needs a rendered pixmap at all ----
                # (tool images come from tables with a desc/photo column;
                #  instruction images come from embedded raster images)
                fitz_page  = fitz_doc[page_index]
                all_imgs   = fitz_page.get_images(full=True)
                needs_tool_render  = bool(raw_tables)
                needs_instr_render = bool(all_imgs)

                page_pil = None
                pw = ph = 0
                if needs_tool_render or needs_instr_render:
                    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
                    page_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    pw, ph = page_pil.size

                # ---- TOOL IMAGE EXTRACTION (from desc/photo tables) ----
                if needs_tool_render and raw_tables:
                    for tbl_obj, tbl_data in zip(found_tables, raw_tables):
                        if not tbl_data:
                            continue

                        desc_idx, photo_idx, header_ri = find_desc_photo_indices(tbl_data)
                        if desc_idx is None:
                            continue

                        col_x0 = tbl_obj.columns[photo_idx].bbox[0]
                        col_x1 = tbl_obj.columns[photo_idx].bbox[2]

                        data_start = header_ri + 1 if header_ri >= 0 else 0

                        for ri in range(data_start, len(tbl_obj.rows)):
                            row_obj  = tbl_obj.rows[ri]
                            row_data = tbl_data[ri] if ri < len(tbl_data) else []
                            if not row_data or desc_idx >= len(row_data):
                                continue

                            label = re.sub(r'\s+', ' ', row_data[desc_idx] or "").strip().upper()
                            if not label or label in ("DESCRIPTION", "TOOL PHOTO", "SR. NO.",
                                                    "SPECIAL TOOL FOR VEHICLE",
                                                    "FRONT SUSPENSION SPECIAL TOOLS"):
                                continue

                            px_x0 = max(0, min(int(col_x0          * SCALE), pw))
                            px_x1 = max(0, min(int(col_x1          * SCALE), pw))
                            px_y0 = max(0, min(int(row_obj.bbox[1] * SCALE), ph))
                            px_y1 = max(0, min(int(row_obj.bbox[3] * SCALE), ph))

                            if px_x1 - px_x0 < 10 or px_y1 - px_y0 < 10:
                                continue

                            safe = re.sub(r'[^\w]', '_', label)[:50]

                            path = os.path.join(
                                BASE_DIR,
                                "images",
                                f"{pdf_short}_p{page_num}_{safe}.png"
                            )
                            page_pil.crop((px_x0, px_y0, px_x1, px_y1)).save(path)

                            ocr_text = ""
                            semantic_label = f"""
                                {label}
                                {ocr_text}
                            """

                            image_data.append({
                                "bike": bike_name,
                                "pdf": pdf_name,
                                "page": page_num,
                                "path": path,
                                "label": label,
                                "ocr": ocr_text,
                                "semantic_label": semantic_label,
                                "image_type": "tool"
                            })

                # ---- INSTRUCTION IMAGE EXTRACTION (floating raster images) ----
                if needs_instr_render and all_imgs:
                    real_imgs = []
                    for img in all_imgs:
                        xref  = img[0]
                        w, h  = img[2], img[3]
                        rects = fitz_page.get_image_rects(xref)
                        if not rects:         continue
                        if w < 100 or h < 90: continue
                        real_imgs.append((xref, rects))

                    if real_imgs:
                        blocks = [b for b in fitz_page.get_text("blocks") if b[6] == 0]

                        for xref, rects in real_imgs:
                            for rect in rects:
                                pos_key = (page_num, round(rect.y0 / 5) * 5, round(rect.y1 / 5) * 5)
                                if pos_key in saved_pos:
                                    continue
                                saved_pos.add(pos_key)

                                nearest = get_best_label(blocks, rect.y0, rect.y1)
                                label   = nearest.strip() if nearest.strip() else f"page {page_num} image"

                                px_x0 = max(0, min(int(rect.x0 * SCALE), pw))
                                px_x1 = max(0, min(int(rect.x1 * SCALE), pw))
                                px_y0 = max(0, min(int(rect.y0 * SCALE), ph))
                                px_y1 = max(0, min(int(rect.y1 * SCALE), ph))

                                if px_x1 - px_x0 < 20 or px_y1 - px_y0 < 20:
                                    continue

                                safe_text = nearest.strip() or f"page_{page_num}_{xref}"
                                safe = re.sub(r'[^\w]', '_', safe_text)[:80]

                                path = os.path.join(
                                    BASE_DIR,
                                    "instr_images",
                                    f"{pdf_short}_p{page_num}_{xref}_{safe}.png"
                                )
                                page_pil.crop((px_x0, px_y0, px_x1, px_y1)).save(path)

                                image_data.append({
                                    "bike": bike_name,
                                    "pdf": pdf_name,
                                    "page": page_num,
                                    "path": path,
                                    "section": get_section(blocks),
                                    "label": label,
                                    "nearest": nearest,
                                    "semantic_label": label,
                                    "y0": rect.y0,
                                    "y1": rect.y1,
                                    "image_type": "instruction"
                                })

        fitz_doc.close()

    #---------------------------------------SEMANTIC SEARCH EMBEDDING---------------------------------------------------------------------------
    
    semantic_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    model=SentenceTransformer("BAAI/bge-base-en-v1.5")

    # Create embeddings for all images
    image_labels = [
        img.get(
            "semantic_label",
            img.get("label", f"Page {img.get('page', 'unknown')} image")
        )
        for img in image_data
    ]

        
    # Generate and store embeddings
    try:
        image_embeddings = model.encode(image_labels,batch_size=64, normalize_embeddings=True).astype("float32")
    except Exception as e:
        print(f"Warning: Could not encode image labels: {e}")
        print(f"Creating fallback embeddings for {len(image_data)} images...")
        image_embeddings = np.zeros((len(image_data), 384), dtype="float32")

    for i, img in enumerate(image_data):
        img["embedding"] = image_embeddings[i]


    #Final text
    final_text = "\n".join(
        [item["text"] for item in text_content]
        + ["\n\n---TABLES---\n\n"]
        + [item["text"] for item in table_content]
        )


    #print text in testextract.txt
    with open("textextract.txt","w",encoding="utf-8") as txt:
        txt.write(final_text)


    #text chunking with image pointer mapping
    chunks = []
    chunk_metadata = []
    chunk_to_images = {}  # Maps chunk index to list of image indices

#-----------------------------SEMANTIC SPLITTER--------------------------------------------
    semantic_splitter = SemanticChunker(
        semantic_embeddings,
        breakpoint_threshold_type="percentile"
    )
#-----------------------------RECURSIVE SPLITTER--------------------------------------------

    recursive_splitter = RecursiveCharacterTextSplitter(      
        chunk_size=800,
        chunk_overlap=50
    )

    images_by_page = {}

    for img_idx, img in enumerate(image_data):

        key = (img["pdf"], img["page"])

        if key not in images_by_page:
            images_by_page[key] = []

        images_by_page[key].append(img_idx)

    for item in text_content:

        pdf_name = item["pdf"]
        page_num = item["page"]
        text = item["text"]

        CHUNKING_METHOD = "semantic"

        if CHUNKING_METHOD == "semantic":
            split_chunks = semantic_splitter.split_text(text)
        else:
            split_chunks = recursive_splitter.split_text(text)

        for chunk in split_chunks:

            chunk = chunk.strip()

            if len(chunk) < 30:
                continue

            chunk_idx = len(chunks)
            chunks.append(chunk)

            chunk_metadata.append({
                "bike": item["bike"],
                "pdf":pdf_name,
                "page": page_num,
                "type": "text",
                
            })
            
            # Map instruction images from new2.pdf to text chunks on same page
            associated_images = [
                img_idx for img_idx in images_by_page.get((pdf_name, page_num), [])
                if "instr_images" in image_data[img_idx].get("path", "")
            ]

            if associated_images:
                chunk_to_images[chunk_idx] = associated_images

    for row in table_content:

        row_text = row["text"].strip()

        if len(row_text) < 20:
            continue

        chunk_idx = len(chunks)
        chunks.append(row_text)

        chunk_metadata.append({
            "bike": row["bike"],
            "pdf": row["pdf"],
            "page": row["page"],
            "type": "table"
        })
        
        # Map table images from new.pdf to table chunks
        associated_images = [
            img_idx for img_idx in images_by_page.get((row["pdf"], row["page"]), [])
            if "instr_images" not in image_data[img_idx].get("path", "")
        ]

        if associated_images:
            chunk_to_images[chunk_idx] = associated_images


    #data embedding
    embeddings = model.encode(
        chunks,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    #storing in faiss
    embeddings=np.array(embeddings).astype("float32")
    dimention=embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index=faiss.IndexFlatIP(dimention)
    start = time.time()
    index.add(embeddings)
    end = time.time()

    print(f"FAISS Ingestion: {(end-start):.4f} sec")

    faiss.write_index(index, os.path.join(BASE_DIR, "vector_store.faiss"))
    index = faiss.read_index( os.path.join(BASE_DIR, "vector_store.faiss"))


    

    print("Loading from:", os.path.abspath("image_data.pkl"))
    print("Exists:", os.path.exists("image_data.pkl"))
    print("Reached save section")
    
    #storing chunks with metadata and image pointers
    with open(os.path.join(BASE_DIR,"chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    with open(os.path.join(BASE_DIR,"chunk_metadata.pkl"), "wb") as f:
        pickle.dump(chunk_metadata, f)

    with open(os.path.join(BASE_DIR,"chunk_to_images.pkl"), "wb") as f:
        pickle.dump(chunk_to_images, f)

    with open(os.path.join(BASE_DIR, "image_data.pkl"), "wb") as f:
        pickle.dump(image_data, f)
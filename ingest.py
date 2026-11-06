"""Indexation multimodale : PDF -> texte / tableaux / images -> PGVector."""
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
import ollama
import pdfplumber
from PIL import Image
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

from common import IMG_DIR, VLM_MODEL, clip_model, conn, init_db, text_model


def table_to_markdown(rows):
    rows = [[(c or "").replace("\n", " ") for c in r] for r in rows if r]
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md)


def extract_images(pdf_path, doc_name):
    """Images intégrées ; si le PDF n'a que des figures vectorielles
    (cas de 'Attention is all you need'), on rend la page en PNG."""
    Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
    out = []
    pdf = fitz.open(pdf_path)
    for pno, page in enumerate(pdf, start=1):
        found = False
        for i, info in enumerate(page.get_images(full=True)):
            pix = fitz.Pixmap(pdf, info[0])
            if pix.width < 100 or pix.height < 100:
                continue
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            p = f"{IMG_DIR}/{doc_name}_p{pno}_{i}.png"
            pix.save(p)
            out.append((pno, p))
            found = True
        if not found and "Figure" in page.get_text():
            p = f"{IMG_DIR}/{doc_name}_p{pno}_render.png"
            page.get_pixmap(dpi=130).save(p)
            out.append((pno, p))
    return out


def caption(path):
    r = ollama.chat(model=VLM_MODEL, messages=[{
        "role": "user",
        "content": "Décris précisément cette figure technique (composants, "
                   "flèches, légendes, texte visible) en français.",
        "images": [path]}])
    return r["message"]["content"]


def ingest(pdf_path):
    pdf_path = str(pdf_path)
    doc_name = Path(pdf_path).stem
    init_db()
    tm, cm = text_model(), clip_model()
    with conn() as c:
        c.execute("DELETE FROM chunks WHERE doc=%s", (doc_name,))
        c.execute("DELETE FROM images WHERE doc=%s", (doc_name,))

        # 1) Texte avec LlamaIndex
        docs = PDFReader().load_data(Path(pdf_path))
        nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64) \
            .get_nodes_from_documents(docs)
        for n in nodes:
            page = int(n.metadata.get("page_label", 0) or 0)
            c.execute(
                "INSERT INTO chunks(doc,page,kind,content,embedding) "
                "VALUES (%s,%s,'text',%s,%s)",
                (doc_name, page, n.get_content(),
                 tm.encode(n.get_content(), normalize_embeddings=True)))

        # 2) Tableaux
        with pdfplumber.open(pdf_path) as pdf:
            for pno, page in enumerate(pdf.pages, start=1):
                for t in page.extract_tables():
                    md = table_to_markdown(t)
                    if md:
                        c.execute(
                            "INSERT INTO chunks(doc,page,kind,content,embedding) "
                            "VALUES (%s,%s,'table',%s,%s)",
                            (doc_name, pno, md,
                             tm.encode(md, normalize_embeddings=True)))

        # 3) Images : CLIP + description textuelle indexée
        for pno, path in extract_images(pdf_path, doc_name):
            cap = caption(path)
            c.execute(
                "INSERT INTO images(doc,page,path,caption,embedding) "
                "VALUES (%s,%s,%s,%s,%s)",
                (doc_name, pno, path, cap,
                 cm.encode(Image.open(path), normalize_embeddings=True)))
            c.execute(
                "INSERT INTO chunks(doc,page,kind,content,meta,embedding) "
                "VALUES (%s,%s,'image_caption',%s,%s,%s)",
                (doc_name, pno, cap, json.dumps({"path": path}),
                 tm.encode(cap, normalize_embeddings=True)))
    return doc_name


if __name__ == "__main__":
    print("Indexé :", ingest(sys.argv[1]))

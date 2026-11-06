"""RAG agentique : l'agent LangChain/LangGraph choisit ses outils de recherche."""
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from common import LLM_MODEL, clip_model, conn, text_model

FOUND_IMAGES = []  # chemins des images retrouvées (affichées par Streamlit)

SYSTEM = (
    "Tu es MaintiDoc, assistant documentaire. Réponds en français UNIQUEMENT "
    "à partir des résultats des outils. Cite toujours la page (ex: [p.3]). "
    "Utilise search_text pour le texte et les tableaux, search_images pour "
    "les figures/schémas/photos. Si l'information est absente, dis-le."
)


@tool
def search_text(query: str) -> str:
    """Cherche des passages de texte et des tableaux du PDF indexé."""
    v = text_model().encode(query, normalize_embeddings=True)
    with conn() as c:
        rows = c.execute(
            "SELECT page, kind, content FROM chunks "
            "ORDER BY embedding <=> %s LIMIT 5", (v,)).fetchall()
    return "\n\n".join(f"[p.{p}] ({k}) {t}" for p, k, t in rows) or "Aucun résultat."


@tool
def search_images(query: str) -> str:
    """Cherche des images/figures du PDF (recherche CLIP) et renvoie leur description."""
    v = clip_model().encode(query, normalize_embeddings=True)
    with conn() as c:
        rows = c.execute(
            "SELECT page, path, caption FROM images "
            "ORDER BY embedding <=> %s LIMIT 2", (v,)).fetchall()
    FOUND_IMAGES.extend(p for _, p, _ in rows)
    return "\n\n".join(f"[p.{pg}] image {p}: {cap}" for pg, p, cap in rows) \
        or "Aucune image."


def build_agent():
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return create_react_agent(llm, [search_text, search_images], prompt=SYSTEM)


def ask(agent, question):
    FOUND_IMAGES.clear()
    res = agent.invoke({"messages": [("user", question)]})
    return res["messages"][-1].content, list(dict.fromkeys(FOUND_IMAGES))

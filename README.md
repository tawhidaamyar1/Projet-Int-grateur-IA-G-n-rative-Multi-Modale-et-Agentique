# MaintiDoc – Partie I : RAG multimodal agentique

## Installation
```bash
docker compose up -d                  # PostgreSQL + pgvector
ollama pull qwen2.5:7b                # LLM (tool calling)
ollama pull qwen2.5vl:7b              # description des images
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Charger `attention_is_all_you_need.pdf`, cliquer sur « Indexer », puis poser :
- « Quelle est la valeur de d_model et du nombre de têtes h ? »
- « Montre-moi le schéma de l'architecture du Transformer. »
- « Compare les scores BLEU du Transformer (big) avec ConvS2S (Tableau 2). »

## Architecture
PDF → LlamaIndex (texte) + pdfplumber (tableaux) + PyMuPDF (images)
→ bge-small (texte) / CLIP (images) + descriptions Qwen2.5-VL → PGVector
→ agent LangGraph (outils search_text / search_images) → Qwen2.5 (Ollama) → Streamlit.

## Limites à mentionner dans la note technique
- Les figures de l'article sont vectorielles : on rend la page en PNG.
- Extraction de tableaux par pdfplumber : fragile sur les mises en page complexes.
- Modèle 7B local : tool calling parfois instable ; réponses en français sur un PDF anglais.
- Les API LangChain/LangGraph évoluent : adapter `create_react_agent` si votre version l'a renommé.

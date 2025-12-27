# Mini RAG — Ask Your Documents

Upload one `.txt` file → ask questions → answers grounded only in that file.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

uvicorn app.main:app --reload

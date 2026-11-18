import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Cargar variables de entorno (.env)
load_dotenv()

# Rutas dinámicas relativas a la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "tramites_suit_boyaca.json"
CHROMA_DIR = BASE_DIR / "chroma_db"


def construir_base_vectorial():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Falta configurar OPENAI_API_KEY en el archivo .env")

    print(f"Leyendo base documental desde: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        tramites = json.load(f)

    documentos = []
    for item in tramites:
        contenido = (
            f"Trámite: {item['titulo']}\n"
            f"Entidad responsable: {item['entidad']}\n"
            f"Requisitos: {item['requisitos']}\n"
            f"Costo: {item['costo']}\n"
            f"Tiempo de respuesta: {item['tiempo_respuesta']}"
        )
        doc = Document(
            page_content=contenido,
            metadata={
                "id": item["id"],
                "titulo": item["titulo"],
                "url": item["url_oficial"],
            },
        )
        documentos.append(doc)

    # Fragmentación (Chunking)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    fragmentos = splitter.split_documents(documentos)
    print(f"Total de fragmentos generados: {len(fragmentos)}")

    # Generación de Embeddings y almacenamiento en ChromaDB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Base de datos vectorial creada exitosamente en: {CHROMA_DIR}")


if __name__ == "__main__":
    construir_base_vectorial()

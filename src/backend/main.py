import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

app = FastAPI(
    title="API Asistente Inteligente Trámites Boyacá (RAG)",
    description="Backend FastAPI + LangChain + ChromaDB para orientación ciudadana SUIT",
    version="1.0.0",
)

# Habilitar CORS para permitir peticiones desde el Frontend (React / GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


SYSTEM_PROMPT = """Eres un asistente virtual oficial de orientación ciudadana para trámites y servicios públicos del departamento de Boyacá, Colombia.
Reglas estrictas de comportamiento:
1. Responde ÚNICAMENTE utilizando la información presente en el CONTEXTO OFICIAL proporcionado abajo.
2. Si el contexto no contiene información suficiente para responder la consulta, indica amablemente que no cuentas con información oficial indexada sobre ese trámite en el SUIT y evita inventar o suponer datos.
3. Redacta en lenguaje claro, empático y estructurado con viñetas para facilitar la comprensión ciudadana.

CONTEXTO OFICIAL (SUIT):
{context}
"""


@app.get("/")
def health_check():
    return {
        "estado": "activo",
        "servicio": "Asistente RAG Trámites Boyacá",
        "nivel_madurez": "TRL 5",
    }


@app.post("/api/chat")
async def chat_rag(payload: QueryRequest):
    pregunta = payload.question.strip()
    if not pregunta:
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")

    if not CHROMA_DIR.exists():
        raise HTTPException(
            status_code=500,
            detail="Base vectorial no encontrada. Ejecuta primero scripts/ingestar_datos.py",
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    # Recuperación semántica con puntaje de relevancia
    resultados = vectorstore.similarity_search_with_relevance_scores(pregunta, k=2)

    # Umbral anti-alucinación: si la similitud es muy baja, no se inventa respuesta
    if not resultados or resultados[0][1] < 0.35:
        return {
            "answer": (
                "Lo siento, no encontré información oficial sobre esa consulta dentro de los "
                "20 trámites de Boyacá indexados en el SUIT. Para evitar datos imprecisos, "
                "te sugiero consultar directamente el portal oficial de Función Pública."
            ),
            "source": {
                "titulo": "Portal Oficial SUIT - Función Pública",
                "url": "https://www.funcionpublica.gov.co/es/suit",
            },
            "similarity": 0,
            "context": "Sin coincidencia semántica relevante en la base vectorial.",
        }

    mejor_doc, score = resultados[0]
    contexto_unido = "\n\n---\n\n".join([doc.page_content for doc, _ in resultados])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    cadena = prompt | llm
    respuesta_ia = cadena.invoke({"context": contexto_unido, "question": pregunta})

    similitud_porcentaje = min(99, max(50, round(score * 100)))

    return {
        "answer": respuesta_ia.content,
        "source": {
            "titulo": mejor_doc.metadata.get("titulo", "Trámite Oficial SUIT"),
            "url": mejor_doc.metadata.get("url", "https://www.funcionpublica.gov.co/es/suit"),
        },
        "similarity": similitud_porcentaje,
        "context": mejor_doc.page_content,
    }

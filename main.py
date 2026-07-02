import faiss
import numpy
from langchain.agents import create_agent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faiss import IndexFlatL2
    from typing import Any
    from langgraph.graph.state import CompiledStateGraph

from sentence_transformers import SentenceTransformer

from numpy import ndarray


def chunkify_text(text: str, chunk_size: int = 500) -> list[str]:
    return [
        text[start:start+chunk_size]
        for start in range(0, len(text), chunk_size)
    ]


def load_doc(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file_obj:
        text: str = file_obj.read()
    
    return text


def create_embeddings(model: SentenceTransformer, data: list[str]) -> ndarray:
    return model.encode(
        inputs=data,
        convert_to_numpy=True,
    )


def build_faiss_index(embeddings: ndarray) -> IndexFlatL2:
    dimension: int = embeddings.shape[1]
    faiss_index: IndexFlatL2 = faiss.IndexFlatL2(dimension)
    faiss_index.add(
        numpy.array(
            object=embeddings, 
            dtype="float32",
        )
    )
    return faiss_index


def build_context(
    faiss_index: IndexFlatL2,
    query_vector: ndarray,
    data: list[str],
) -> str:
    query_vector_reshaped: ndarray = numpy.array(query_vector, dtype="float32").reshape(1, -1)
    _, indices = faiss_index.search(
        x=query_vector_reshaped,
        k=3,
    )

    retrieved: list[str] = [
        data[i]
        for i in indices[0]
    ]

    return "\n\n".join(retrieved)
    

def build_prompt(context: str, question: str) -> str:
    return (
        "You are a helpful assistant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


if __name__ == "__main__":
    
    loaded_text: str = load_doc(
        file_path="./text.txt",
    )

    chunks: list[str] = chunkify_text(
        text=loaded_text,
        chunk_size=200,
    )

    sentence_transformer: SentenceTransformer = SentenceTransformer(
        model_name_or_path="all-MiniLM-L6-v2",
    )

    embeddings: ndarray = create_embeddings(
        model=sentence_transformer,
        data=chunks,
    )

    faiss_index: IndexFlatL2 = build_faiss_index(
        embeddings=embeddings,
    )

    questions: list[str] = [
        "How many vacation days do employees get?",
    ]
    
    query_vector: ndarray = create_embeddings(
        model=sentence_transformer,
        data=questions,
    )

    context: str = build_context(
        faiss_index=faiss_index,
        query_vector=query_vector,
        data=chunks,
    )

    prompt: str = build_prompt(
        context=context,
        question=questions[0],
    )

    print(prompt)

    agent: CompiledStateGraph = create_agent(
        model="ollama:llama3.2",
        system_prompt="You are a helpful assistant"
    )

    result: dict[str, Any] = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }
    )

    print(result["messages"][-1].content)

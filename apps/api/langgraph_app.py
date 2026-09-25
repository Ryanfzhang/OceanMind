"""Minimal API entry point for the LangGraph backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.langgraph_dataset import router as dataset_router
from apps.api.langgraph_query import router as query_router
from apps.api.langgraph_results import router as results_router
from apps.api.langgraph_visualize import router as visualize_router


app = FastAPI(title="OceanMind API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset_router)
app.include_router(query_router)
app.include_router(results_router)
app.include_router(visualize_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

import os
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from urllib.parse import urlparse

# from mem_t.neuro_mem import memory_model
from mem_t.memo_tra_model import memory_model

load_dotenv()

MEM0_BASE_URL = os.getenv("MEM0_BASE_URL")
port = urlparse(MEM0_BASE_URL).port or 8123

app = FastAPI()


# Pydantic models for requests
class AddMemoryRequest(BaseModel):
    model_config = ConfigDict(extra='allow')

    message: Union[str, List[Dict[str, str]]] = Field(validation_alias="messages")
    user_id: str
    version: Optional[str] = "v2"
    metadata: Optional[Dict] = None
    enable_graph: Optional[bool] = False


class SearchMemoryRequest(BaseModel):
    model_config = ConfigDict(extra='allow')

    query: str
    user_id: str
    top_k: Optional[int] = 10
    filter_memories: Optional[bool] = False
    enable_graph: Optional[bool] = False
    output_format: Optional[str] = "v1.0"


class DeleteAllRequest(BaseModel):
    user_id: str


class UpdateProjectRequest(BaseModel):
    model_config = ConfigDict(extra='allow')

    custom_instructions: str


@app.post("/v1/memories/")
async def add_memory(request: AddMemoryRequest):
    try:
        if isinstance(request.message, list):
            # Concatenate the contents
            # text = " ".join([msg.get("content", str(msg)) for msg in request.message])
            text = request.message[-1].get("content", "")
        else:
            text = request.message
        key = memory_model.add_memory(
            text=text, user_id=request.user_id, meta=request.metadata
        )
        return {"message": "Memory added successfully", "key": key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memories/search/")
async def search_memory(request: SearchMemoryRequest):
    try:
        results = memory_model.query(
            query_text=request.query, user_id=request.user_id, top_k=request.top_k or 10
        )
        # Format results like mem0
        memories = []
        for score, item in results:
            memories.append(
                {
                    "memory": item.text,
                    "metadata": item.meta or {},
                    "score": round(score, 2),
                }
            )
        if request.enable_graph:
            # HMM doesn't support graph, so return empty relations
            return {"results": memories, "relations": []}
        else:
            return memories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories/")
async def delete_all_memories(
    user_id: str, org_id: Optional[str] = None, project_id: Optional[str] = None
):
    try:
        deleted_count = memory_model.delete_all(user_id)
        return {
            "message": f"Deleted {deleted_count} memories for user {user_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/project")
async def update_project(request: UpdateProjectRequest):
    # For now, just acknowledge - HMM doesn't use custom instructions
    return {"message": "Project updated successfully"}

@app.patch("/api/v1/orgs/organizations/{org_id}/projects/{project_id}/")
async def patch_project(org_id: str, project_id: str, request: UpdateProjectRequest):
	# For now, just acknowledge - HMM doesn't use custom instructions
	return {"message": "Project patched successfully"}

@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "HMM Memory Model"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)

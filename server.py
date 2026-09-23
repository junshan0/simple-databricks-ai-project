# server.py
import os
from mcp.server.fastmcp import FastMCP
from databricks.sdk import WorkspaceClient
​
mcp = FastMCP("video-knowledge")
w   = WorkspaceClient()
WAREHOUSE_ID = os.environ["DATABRICKS_WAREHOUSE_ID"]  # from app resource
​
def _sql(query: str, params: list | None = None):
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=query,
        parameters=params or [],
    )
    return resp.result.data_array if resp.result else []
​
@mcp.tool()
def search_video(query: str, k: int = 4) -> list[dict]:
    """Semantically search all video chunks; returns top-k with timestamps."""
    rows = _sql("SELECT * FROM video_ai.ai.search_video(:q, :k)",
                [{"name": "q", "value": query},
                 {"name": "k", "value": str(k)}])
    cols = ["chunk_id", "video_id", "start_time", "end_time", "caption", "topic"]
    return [dict(zip(cols, r)) for r in rows]
​
@mcp.tool()
def summarize_video(video_id: str) -> str:
    """Summarize a video from its chunk captions."""
    rows = _sql("SELECT video_ai.ai.summarize_video(:v)",
                [{"name": "v", "value": video_id}])
    return rows[0][0] if rows else ""
​
# add get_video_metadata / get_video_chunk / find_topic / query_video_transcript similarly
​
if __name__ == "__main__":
    mcp.run(transport="streamable-http",
            host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
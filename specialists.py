# specialists.py
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from mlflow.deployments import get_deploy_client
import json

w   = WorkspaceClient()
llm = get_deploy_client("databricks")
LLM = "databricks-meta-llama-3-3-70b-instruct"

mcp = DatabricksMCPClient(
    server_url=f"{w.config.host}/api/2.0/mcp/functions/video_ai/ai",
    workspace_client=w,
)
ALL_TOOLS = {t.name: t for t in mcp.list_tools()}

def _openai_tools(names):
    return [{
        "type": "function",
        "function": {"name": ALL_TOOLS[n].name,
                     "description": ALL_TOOLS[n].description,
                     "parameters": ALL_TOOLS[n].input_schema},
    } for n in names]

def run_agent(system_prompt: str, tool_names: list[str], task: str) -> str:
    """A minimal tool-calling loop for one specialist."""
    tools = _openai_tools(tool_names)
    msgs  = [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": task}]
    for _ in range(4):  # cap tool-call rounds
        resp = llm.predict(endpoint=LLM, inputs={"messages": msgs, "tools": tools})
        msg  = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg["content"]
        msgs.append(msg)
        for c in calls:
            out = mcp.call_tool(c["function"]["name"],
                                json.loads(c["function"]["arguments"]))
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": str(out)})
    return msgs[-1].get("content", "")

# --- The three specialists (name resolution: adjust to your tool naming) ---
def video_search_agent(task):
    return run_agent(
        "You find relevant video segments. Return video_id, timestamps, and captions. "
        "Do not summarize; just retrieve.",
        ["video_ai__ai__search_video", "video_ai__ai__query_video_transcript"],
        task)

def data_agent(task):
    return run_agent(
        "You extract structured facts about videos (instructors, topics, categories). "
        "Use metadata tools; return concise structured findings.",
        ["video_ai__ai__get_video_metadata", "video_ai__ai__find_topic"],
        task)

def summary_agent(task):
    return run_agent(
        "You synthesize and compare content. Produce a clear summary and cite "
        "(video_id, start-end) for each claim.",
        ["video_ai__ai__summarize_video", "video_ai__ai__get_video_chunk"],
        task)
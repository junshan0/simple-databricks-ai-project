# rag_mcp_agent.py  (extends the Section 5 agent)
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from mlflow.deployments import get_deploy_client

w        = WorkspaceClient()
mcp      = DatabricksMCPClient(
    server_url=f"{w.config.host}/api/2.0/mcp/functions/video_ai/ai",
    workspace_client=w,
)
llm      = get_deploy_client("databricks")
LLM      = "databricks-claude-3-7-sonnet"

def to_openai_tools(mcp_tools):
    return [{
        "type": "function",
        "function": {"name": t.name, "description": t.description,
                     "parameters": t.inputSchema},
    } for t in mcp_tools]

def chat(question: str):
    tools = mcp.list_tools()
    msgs  = [{"role": "user", "content": question}]
    resp  = llm.predict(endpoint=LLM,
                        inputs={"messages": msgs, "tools": to_openai_tools(tools)})
    msg   = resp["choices"][0]["message"]

    # If the model asked to call a tool, execute it via MCP and feed the result back
    for call in msg.get("tool_calls", []) or []:
        import json
        out = mcp.call_tool(call["function"]["name"],
                            json.loads(call["function"]["arguments"]))
        msgs += [msg, {"role": "tool", "tool_call_id": call["id"],
                       "content": str(out)}]
    final = llm.predict(endpoint=LLM, inputs={"messages": msgs})
    return final["choices"][0]["message"]["content"]

print(chat("Summarize what lesson03 covers and cite timestamps."))
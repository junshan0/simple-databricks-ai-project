# supervisor.py
from mlflow.deployments import get_deploy_client
import json
from specialists import video_search_agent, data_agent, summary_agent

llm = get_deploy_client("databricks")
LLM = "databricks-meta-llama-3-3-70b-instruct"

SPECIALISTS = {
    "video_search_agent": video_search_agent,
    "data_agent":         data_agent,
    "summary_agent":      summary_agent,
}

SUPERVISOR_TOOLS = [
    {"type": "function", "function": {
        "name": "video_search_agent",
        "description": "Find relevant video segments for a query. Returns video_ids, timestamps, captions.",
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}}, "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "data_agent",
        "description": "Extract structured facts (instructors, topics, categories) about given videos.",
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}}, "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "summary_agent",
        "description": "Summarize or compare content across videos, with citations.",
        "parameters": {"type": "object",
                       "properties": {"task": {"type": "string"}}, "required": ["task"]}}},
]

SUPERVISOR_PROMPT = """You are a supervisor coordinating three specialist agents.
Break the user's request into sub-tasks and call the right specialists (you may call several,
in sequence, feeding one's output into the next). When you have enough, write the final answer.
Always keep the (video_id, start-end) citations the specialists provide."""

def supervise(user_request: str) -> str:
    msgs = [{"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user",   "content": user_request}]
    for _ in range(6):  # cap orchestration rounds
        resp = llm.predict(endpoint=LLM,
                           inputs={"messages": msgs, "tools": SUPERVISOR_TOOLS})
        msg  = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg["content"]           # final synthesized answer
        msgs.append(msg)
        for c in calls:
            args = json.loads(c["function"]["arguments"])
            result = SPECIALISTS[c["function"]["name"]](args["task"])
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return msgs[-1].get("content", "")

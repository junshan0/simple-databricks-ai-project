# rag_agent.py
import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient

VS_ENDPOINT  = "video_ai_vs_endpoint"
VS_INDEX     = "video_ai.silver.video_chunk_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

class VideoRAGAgent(ChatAgent):
    def __init__(self):
        self.w      = WorkspaceClient()
        self.client = get_deploy_client("databricks")

    def _retrieve(self, question, k=4):
        res  = self.w.vector_search_indexes.query_index(
            index_name=VS_INDEX,
            query_text=question,
            columns=["chunk_id", "video_id",
                     "start_time", "end_time", "caption", "topic"],
            num_results=k,
        )
        cols = [c.name for c in res.manifest.columns]
        return [dict(zip(cols, r)) for r in res.result.data_array]

    def predict(self, messages, context=None, custom_inputs=None) -> ChatAgentResponse:
        question = messages[-1].content
        chunks   = self._retrieve(question)
        ctx = "\n\n".join(
            f"video={c['video_id']} {c['start_time']}-{c['end_time']}: {c['caption']}"
            for c in chunks
        )
        resp = self.client.predict(
            endpoint=LLM_ENDPOINT,
            inputs={"messages": [
                {"role": "system", "content":
                    "Answer only from context; end with (video <id>, <start>-<end>)."},
                {"role": "user", "content": f"Q: {question}\n\nContext:\n{ctx}"},
            ]},
        )
        text = resp["choices"][0]["message"]["content"]
        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=text, id="1")]
        )

from mlflow.models import set_model
set_model(VideoRAGAgent())
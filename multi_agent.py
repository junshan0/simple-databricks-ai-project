# multi_agent.py
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from mlflow.models import set_model
from supervisor import supervise

class VideoMultiAgent(ChatAgent):
    def predict(self, messages, context=None, custom_inputs=None) -> ChatAgentResponse:
        answer = supervise(messages[-1].content)
        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=answer, id="1")])

set_model(VideoMultiAgent())
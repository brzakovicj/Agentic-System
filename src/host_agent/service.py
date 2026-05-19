import json
from uuid import uuid4
from langchain_core.messages import SystemMessage
from src.a2a_services.a2a_client import A2A_Client
from src.prompts.prompt_manager import PromptManager
from src.utils.llm_factory import LLMFactory

SCHOLAR_URL = "http://127.0.0.1:9001"
AGENDA_URL = "http://127.0.0.1:9002"

CAPABILITIES_MESSAGE = """\
## 👋 Here's what I can help you with:

**🔍 Research**
Ask me to research any topic and I'll gather comprehensive information for you.
> *"Research the causes of World War I"*

**📝 Study Notes**
I'll research a topic and generate clean, structured notes ready for revision.
> *"Generate study notes on the water cycle"*

**📅 Exam Schedule**
Ask about your upcoming exams and I'll pull up your schedule.
> *"What exams do I have this week?"*

**🧠 Topic Q&A**
Ask any question about your subjects and I'll give you a thorough answer.
> *"Explain how mitosis works"*

---
*Try rephrasing your question, or pick one of the options above!*"""

SOMETHING_WENT_WRONG_MESSAGE = """\
## ⚠️ Something went wrong on my end.

I wasn't able to route your request correctly. Here's what I *can* help you with — try rephrasing your question around one of these:

**🔍 Research** — *"Research quantum computing for me"*
**📝 Study Notes** — *"Generate study notes on the French Revolution"*
**📅 Exam Schedule** — *"What exams do I have coming up?"*
**🧠 Topic Q&A** — *"Explain Newton's laws of motion"*
"""

class HostAgentService:

    def __init__(self):

        self.client = A2A_Client(
            known_agent_urls=[
                SCHOLAR_URL,
                AGENDA_URL
            ]
        )

        llm_factory = LLMFactory.initialize()

        self.llm = llm_factory.get_remote_llm()

        self.prompt_manager = PromptManager()

    async def process_message(self, user_input: str) -> str:

        result = await self.client.a2a_list_discovered_agents()

        agent_cards = ""

        if result["status"] == "success":
            for agent in result["agents"]:
                agent_cards += json.dumps(agent, indent=2) + "\n"

        prompt = self.prompt_manager.get(
            "host_supervisor_prompt",
            user_input=user_input,
            agent_cards=agent_cards
        )

        response = await self.llm.ainvoke([
            SystemMessage(content=prompt)
        ])

        selected_agent = response.content.strip()

        if selected_agent == "NONE":
            return CAPABILITIES_MESSAGE
        
        if selected_agent not in (SCHOLAR_URL, AGENDA_URL):
            return SOMETHING_WENT_WRONG_MESSAGE

        message_id = str(uuid4())

        result = await self.client.a2a_send_message(
            message_text=user_input,
            target_agent_url=selected_agent,
            message_id=message_id,
        )

        if result["status"] == "error":
            return f"Error: {result['error']}"

        response_data = result["response"]

        ######################### NE OBRADJUJU SE UPDATE-OVI

        # artifact
        if response_data["type"] == "artifact":

            artifact = response_data["data"]

            parts = artifact.get("parts", [])

            if parts:
                return parts[0].get("text", "No text response.")

        # direct message
        if response_data["type"] == "message":

            message = response_data["data"]

            parts = message.get("parts", [])

            if parts:
                return parts[0].get("text", "No text response.")

        return "No response received."
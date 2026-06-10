from src.host_agent.service import HostAgentService

class HostAgent:

    def __init__(self):
        self.service = HostAgentService()

    async def astream(
        self,
        query: str,
        context_id: str,
    ):
        async for event in self.service.process_message_stream(
            query,
            context_id,
        ):

            metadata = event.get("metadata") or {}

            yield {
                "is_task_complete": event["type"] == "final",
                "require_user_input": event["type"] == "input_required",
                "content": event["content"],
                "call_type": metadata.get("call_type"),
                "node_id": metadata.get("node_id"),
                "node_name": metadata.get("node_name"),
                "node_status": metadata.get("node_status"),
                "parent_id": metadata.get("parent_id"),
            }
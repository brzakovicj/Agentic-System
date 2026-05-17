import asyncio
import httpx
import logging
from typing import Any
from uuid import uuid4
from a2a.client import ClientConfig, A2ACardResolver, ClientFactory
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
    TaskPushNotificationConfig
)
from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300

class A2A_Client:

    def __init__(
        self,
        known_agent_urls: list[str] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        webhook_url: str | None = None,
        webhook_token: str | None = None,
    ):
        """
        Initialize A2A client tool provider.

        Args:
            known_agent_urls: List of A2A agent URLs to use (defaults to None)
            timeout: Timeout for HTTP operations in seconds (defaults to 300)
            webhook_url: Optional webhook URL for push notifications
            webhook_token: Optional authentication token for webhook notifications
        """

        self._timeout = timeout
        self._known_agent_urls: list[str] = known_agent_urls or []
        self._discovered_agents: dict[str, AgentCard] = {}
        self._httpx_client: httpx.AsyncClient | None = None
        self._client_factory: ClientFactory | None = None
        self._initial_discovery_done: bool = False

        # Push notification configuration
        self._webhook_url = webhook_url
        self._webhook_token = webhook_token
        self._push_config: TaskPushNotificationConfig | None = None

        if self._webhook_url and self._webhook_token:
            self._push_config = TaskPushNotificationConfig(
                id=f"langgraph-webhook-{uuid4().hex[:8]}", 
                url=self._webhook_url, 
                token=self._webhook_token
            )

    async def _ensure_httpx_client(self) -> httpx.AsyncClient:
        """Ensure the shared HTTP client is initialized."""

        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
            )
        return self._httpx_client
    
    async def _ensure_client_factory(self) -> ClientFactory:
        """Ensure the ClientFactory is initialized."""

        if self._client_factory is None:
            httpx_client = await self._ensure_httpx_client()
            config = ClientConfig(
                httpx_client=httpx_client,
                streaming=True,
                push_notification_config=self._push_config if self._push_config else None,
            )
            self._client_factory = ClientFactory(config)
        return self._client_factory
    
    async def _create_a2a_card_resolver(self, url: str) -> A2ACardResolver:
        """Create a new A2A card resolver for the given URL."""

        httpx_client = await self._ensure_httpx_client()
        logger.info(f"A2ACardResolver created for {url}")
        return A2ACardResolver(httpx_client=httpx_client, base_url=url)
    
    async def _discover_known_agents(self) -> None:
        """Discover all agents provided during initialization."""

        async def _discover_agent_with_error_handling(url: str):
            """Helper method to discover an agent with error handling."""
            try:
                await self._discover_agent_card(url)
            except Exception as e:
                logger.error(f"Failed to discover agent at {url}: {e}")

        tasks = [_discover_agent_with_error_handling(url) for url in self._known_agent_urls]
        if tasks:
            await asyncio.gather(*tasks)

        self._initial_discovery_done = True

    async def _discover_agent_card(self, url: str) -> AgentCard:
        """Internal method to discover and cache an agent card."""

        if url in self._discovered_agents:
            return self._discovered_agents[url]

        resolver = await self._create_a2a_card_resolver(url)
        agent_card = await resolver.get_agent_card()
        self._discovered_agents[url] = agent_card
        logger.info(f"Successfully discovered and cached agent card for {url}")

        return agent_card
    
    async def _ensure_discovered_known_agents(self) -> None:
        """Ensure initial discovery of agent URLs from constructor has been done."""
        
        if not self._initial_discovery_done and self._known_agent_urls:
            await self._discover_known_agents()

    async def a2a_discover_agent(self, url: str) -> dict[str, Any]:
        """
        Discover an A2A agent and return its agent card with capabilities.

        This function fetches the agent card from the specified A2A agent URL
        and caches it for future use.

        Args:
            url: The base URL of the A2A agent to discover

        Returns:
            dict: Discovery result including:
                - success: Whether the operation succeeded
                - agent_card: The full agent card data (if successful)
                - error: Error message (if failed)
                - url: The agent URL that was queried
        """
        try:
            await self._ensure_discovered_known_agents()
            agent_card = await self._discover_agent_card(url)
            return {
                "status": "success",
                "agent_card": agent_card.model_dump(mode="python", exclude_none=True),
                "url": url,
            }
        except Exception as e:
            logger.exception(f"Error discovering agent card for {url}")
            return {
                "status": "error",
                "error": str(e),
                "url": url,
            }
        
    async def a2a_list_discovered_agents(self) -> dict[str, Any]:
        """
        List all discovered A2A agents and their capabilities.

        Returns:
            dict: Information about all discovered agents including:
                - success: Whether the operation succeeded
                - agents: List of discovered agents with their details
                - total_count: Total number of discovered agents
        """

        try:
            await self._ensure_discovered_known_agents()
            agents = [
                agent_card.model_dump(mode="python", exclude_none=True)
                for agent_card in self._discovered_agents.values()
            ]
            return {
                "status": "success",
                "agents": agents,
                "total_count": len(agents),
            }
        except Exception as e:
            logger.exception("Error listing discovered agents")
            return {
                "status": "error",
                "error": str(e),
                "total_count": 0,
            }
        
    async def a2a_send_message(
        self, message_text: str, target_agent_url: str, message_id: str | None = None
    ) -> dict[str, Any]:
        """
        Send a message to a specific A2A agent and return the response.

        Args:
            message_text: The message content to send to the agent
            target_agent_url: The URL of the target A2A agent
            message_id: Optional message ID for tracking (generates UUID if not provided)

        Returns:
            dict: Response data including:
                - success: Whether the message was sent successfully
                - response: The agent's response data (if successful)
                - error: Error message (if failed)
                - message_id: The message ID used
                - target_agent_url: The agent URL that was contacted
        """
        client = None
        try:
            await self._ensure_discovered_known_agents()

            # Get the agent card and create client using factory
            agent_card = await self._discover_agent_card(target_agent_url)
            client_factory = await self._ensure_client_factory()
            client = client_factory.create(agent_card)

            if message_id is None:
                message_id = str(uuid4())

            message = Message(
                message_id=message_id,
                role=Role.ROLE_USER,
                parts=[Part(text=message_text)],
            )

            request = SendMessageRequest(message=message)

            logger.info(f"Sending message to {target_agent_url}")

            async for event in client.send_message(request):
                print(f'Event fields: {[field.name for field, _ in event.ListFields()]}')
                print(f'Event: {event}')
                
                # --- Direktna poruka (agent odgovorio bez taska) ---
                if event.HasField('message'):
                    return {
                        "status": "success",
                        "response": {
                            "type": "message",
                            "data": MessageToDict(event.message, preserving_proto_field_name=True),
                        },
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                    }
                
                # --- Task kreiran ---
                if event.HasField('task'):
                    task_data = MessageToDict(event.task, preserving_proto_field_name=True)
                    logger.info(f"Task data {task_data}")
                
                # --- Status update ---
                elif event.HasField('status_update'):
                    state = event.status_update.status.state
                    state_name = TaskState.Name(state)

                    if state_name in (
                        'TASK_STATE_COMPLETED',
                        'TASK_STATE_FAILED',
                        'TASK_STATE_CANCELED',
                        'TASK_STATE_REJECTED',
                    ):
                        return {
                            "status": "success" if state_name == 'TASK_STATE_COMPLETED' else "error",
                            "response": {
                                "type": "status_update",
                                "state": state_name,
                                "data": MessageToDict(event.status_update, preserving_proto_field_name=True),
                            },
                            "message_id": message_id,
                            "target_agent_url": target_agent_url,
                        }
                
                # --- Artifact (finalni rezultat) ---
                elif event.HasField('artifact_update'):
                    return {
                        "status": "success",
                        "response": {
                            "type": "artifact",
                            "data": MessageToDict(event.artifact_update.artifact, preserving_proto_field_name=True),
                        },
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                    }
                
            return {
                "status": "error",
                "error": "No response received from agent",
                "message_id": message_id,
                "target_agent_url": target_agent_url,
            }

        except Exception as e:
            logger.exception(f"Error sending message to {target_agent_url}")
            return {
                "status": "error",
                "error": str(e),
                "message_id": message_id,
                "target_agent_url": target_agent_url,
            }
        finally:
            if client is not None:
                await client.close()

###############################################################################
# 'Tell me something about Covid 19.'
# 'When is the next Internet of Things exam?'

async def main() -> None:
    """Run the A2A client."""
    SCHOLAR_URL: str = "http://127.0.0.1:9001"
    AGENDA_URL: str = "http://127.0.0.1:9002"
    

    client = A2A_Client(
        known_agent_urls=[SCHOLAR_URL, AGENDA_URL],
    )

    while True:
        try:
            loop = asyncio.get_running_loop()
            user_input = await loop.run_in_executor(None, input, 'You: ')
        except KeyboardInterrupt:
            break

        if user_input.lower() in ('quit', 'exit'):
            break
        if not user_input.strip():
            continue

        try:
            message_id = str(uuid4())

            await client.a2a_send_message(
                message_text=user_input,
                target_agent_url=SCHOLAR_URL,
                message_id=message_id,
            )
        except Exception as e:
            print(f'Error communicating with agent: {e}')

if __name__ == '__main__':
    asyncio.run(main())
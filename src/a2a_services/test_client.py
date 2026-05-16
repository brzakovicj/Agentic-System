import httpx
import asyncio

from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import display_agent_card, new_text_message
from a2a.types.a2a_pb2 import (
    Role,
    SendMessageRequest,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


async def main() -> None:
    # --8<-- [start:A2ACardResolver]
    scholar_base_url = 'http://127.0.0.1:9001'
    agenda_base_url = 'http://127.0.0.1:9002'

    async with httpx.AsyncClient() as httpx_client:
        # Initialize A2ACardResolver
        scholar_resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=scholar_base_url,
            # agent_card_path uses default
        )

        agenda_resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=agenda_base_url,
            # agent_card_path uses default
        )

        # --8<-- [end:A2ACardResolver]

        #######################################################################

        print(
            f'Attempting to fetch public agent card from: {scholar_base_url}{AGENT_CARD_WELL_KNOWN_PATH}'
        )
        scholar_public_card = await scholar_resolver.get_agent_card()
        print('\nSuccessfully fetched public agent card:')
        display_agent_card(scholar_public_card)

        print(
            f'Attempting to fetch public agent card from: {agenda_base_url}{AGENT_CARD_WELL_KNOWN_PATH}'
        )
        agenda_public_card = await agenda_resolver.get_agent_card()
        print('\nSuccessfully fetched public agent card:')
        display_agent_card(agenda_public_card)

    #######################################################################

    print('\n--- SCHOLAR Streaming Call ---')
    # --8<-- [start:message_stream]
    streaming_config = ClientConfig(streaming=True)
    scholar_streaming_client = await create_client(
        agent=scholar_public_card, 
        client_config=streaming_config
    )
    print('\nStreaming Client initialized.')

    message = new_text_message('Tell me something about Covid 19.', role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)
    streaming_response = scholar_streaming_client.send_message(request)

    async for chunk in streaming_response:
        print('Response chunk:')
        print(chunk)
    # --8<-- [end:message_stream]

    await scholar_streaming_client.close()

    print('\n--- AGENDA Streaming Call ---')
    streaming_config = ClientConfig(streaming=True)
    agenda_streaming_client = await create_client(
        agent=agenda_public_card, 
        client_config=streaming_config
    )
    print('\nStreaming Client initialized.')

    message = new_text_message('When is the next Internet of Things exam?', role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)
    streaming_response = agenda_streaming_client.send_message(request)

    async for chunk in streaming_response:
        print('Response chunk:')
        print(chunk)
    # --8<-- [end:message_stream]

    await agenda_streaming_client.close()


if __name__ == '__main__':
    asyncio.run(main())
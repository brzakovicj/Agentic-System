import asyncio
import httpx
from uuid import uuid4
from a2a.client import create_client, ClientConfig
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)

def build_send_request(content: str) -> SendMessageRequest:
    return SendMessageRequest(
        message=Message(
            message_id=str(uuid4()),
            role=Role.ROLE_USER,
            parts=[Part(text=content)],
        )
    )

async def call_agent(base_url: str, question: str, label: str):
    print(f'\n--- {label} ---')

    httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
    )

    client = await create_client(
        agent=base_url,
        client_config=ClientConfig(httpx_client=httpx_client),
    )

    try:
        send_request = build_send_request(question)

        async for chunk in client.send_message(send_request):
            print(f'Chunk fields: {[f.name for f, _ in chunk.ListFields()]}')
            print(f'Chunk: {chunk}')

            for fd, val in chunk.ListFields():
                if hasattr(val, 'parts'):
                    for p in val.parts:
                        if p.text:
                            print(f'  >> {p.text}')
                if hasattr(val, 'state'):
                    if val.state in (
                        TaskState.TASK_STATE_COMPLETED,
                        TaskState.TASK_STATE_FAILED,
                        TaskState.TASK_STATE_CANCELLED,
                    ):
                        print(f'  >> Završeno: {val.state}')
                        return

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        await client.close()
        await httpx_client.aclose()


async def main():
    await call_agent('http://127.0.0.1:9001', 'Tell me something about Covid 19.', 'SCHOLAR')
    await call_agent('http://127.0.0.1:9002', 'When is the next Internet of Things exam?', 'AGENDA')


if __name__ == '__main__':
    asyncio.run(main())
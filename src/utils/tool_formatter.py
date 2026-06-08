import json
from src.utils.llm_factory import LLMFactory

async def llm_describe_tool_call(tc: dict) -> str:
    name: str = tc.get("name", "unknown")
    args: dict = tc.get("args", {})

    try:
        llm = LLMFactory.get_instance().get_local_llm()
        prompt = (
            f"Tool name: {name}\n"
            f"Arguments: {json.dumps(args, ensure_ascii=False)}\n\n"
            "Generate a short first-person progress message describing what I am doing "
            "with this tool call. Use the argument values when relevant. "
            "Start the message with exactly one relevant emoji. "
            "Keep it natural and specific. Maximum 80 characters. "
            "Return only the message text. Do not use quotes or ending punctuation."
        )
        response = await llm.ainvoke(prompt)
        return f"{response.content.strip()}"
    except Exception:
        return f"{name} — {json.dumps(args, ensure_ascii=False)}"
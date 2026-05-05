import sys
import logging
from enum import Enum
from typing import List, Optional
from langchain.tools import BaseTool
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

class ModelTier(Enum):
    LOCAL = "local"
    REMOTE = "remote"


# class FallbackChatModel(BaseChatModel):
#     """Wrapper koji pokušava remote, pa pada na local pri grešci."""
    
#     primary: BaseChatModel
#     fallback: BaseChatModel

#     @property
#     def _llm_type(self) -> str:
#         return "fallback-chat-model"

#     def _generate(self, messages, stop=None, run_manager=None, **kwargs):
#         try:
#             return self.primary._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
#         except Exception as e:
#             logger.warning(f"Remote model failed ({e}), falling back to local.")
#             return self.fallback._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

#     async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
#         try:
#             return await self.primary._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
#         except Exception as e:
#             logger.warning(f"Remote model failed ({e}), falling back to local.")
#             return await self.fallback._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


class LLMFactory:
    _instance: Optional["LLMFactory"] = None

    def __init__(self, mode: ModelTier = ModelTier.REMOTE):
        if LLMFactory._instance is not None:
            raise RuntimeError("Koristi LLMFactory.get_instance() umesto direktnog konstruktora.")
        self.mode = mode
        self._local_llm: Optional[BaseChatModel] = None
        self._remote_llm: Optional[BaseChatModel] = None
        LLMFactory._instance = self

    @classmethod
    def initialize(cls, argv: list[str] = None) -> "LLMFactory":
        """Poziva se JEDNOM na startu programa."""
        if cls._instance is not None:
            raise RuntimeError("LLMFactory je već inicijalizovan.")
        argv = argv or sys.argv[1:]
        mode = ModelTier.REMOTE
        if "--mode" in argv:
            idx = argv.index("--mode")
            try:
                mode = ModelTier(argv[idx + 1])
            except (IndexError, ValueError):
                logger.warning("Invalid --mode value, defaulting to 'remote'.")
        return cls(mode=mode)

    @classmethod
    def get_instance(cls) -> "LLMFactory":
        """Poziva se u node-ovima — uvek vraća istu instancu."""
        if cls._instance is None:
            raise RuntimeError("LLMFactory nije inicijalizovan. Pozovi LLMFactory.initialize() na startu.")
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Private builders                                                    #
    # ------------------------------------------------------------------ #

    def _build_local(self) -> BaseChatModel:
        if self._local_llm is None:
            self._local_llm = ChatOllama(
                model="llama3.2:3b",
                temperature=0,
            )
        return self._local_llm

    def _build_remote(self) -> BaseChatModel:
        # if self._remote_llm is None:
        #     self._remote_llm = ChatOpenAI(
        #         base_url=os.getenv("REMOTE_SERVER_URL"),
        #         api_key="dummy",
        #         model="gemma4:26b",
        #         temperature=0,
        #     )
        # return self._remote_llm
    
        self._remote_llm = ChatOpenAI(
                base_url=os.getenv("REMOTE_SERVER_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gemma4:26b",
                temperature=0,
                default_headers={
                    "Content-Type": "application/json",
                    "Authorization": os.getenv("REMOTE_SERVER_AUTH_TOKEN"),
                }
            )
        
        return self._remote_llm

    # ------------------------------------------------------------------ #
    #  Public accessors                                                    #
    # ------------------------------------------------------------------ #

    def get_local_llm(self) -> BaseChatModel:
        """Uvek vraća lokalni model — za lightweight node-ove."""
        return self._build_local()

    def get_remote_llm(self) -> BaseChatModel:
        """Uvek vraća remote model — bez fallbacka."""
        return self._build_remote()

    def get_base_llm(self) -> BaseChatModel:
        """
        Vraća model prema --mode argumentu.
        Ako je remote, automatski pada na local pri grešci (situacija 2).
        """
        if self.mode == ModelTier.LOCAL:
            return self._build_local()

        return self._build_remote()
    
        # Remote sa fallbackom
        # return FallbackChatModel(
        #     primary=self._build_remote(),
        #     fallback=self._build_local(),
        # )
    
    # ------------------------------------------------------------------ #
    #  Bound variants                                                      #
    # ------------------------------------------------------------------ #

    def get_tool_llm(self, tier: ModelTier, tools: List[BaseTool] = []) -> BaseChatModel:
        """
        tier=ModelTier.LOCAL  → lokalni model sa alatima (lightweight node)
        tier=ModelTier.REMOTE → remote model sa alatima (supervisor)
        tier=None             → get_base_llm() logika
        """
        if tier == ModelTier.LOCAL:
            llm = self._build_local()
        elif tier == ModelTier.REMOTE:
            llm = self._build_remote()

        return llm.bind_tools(tools)

    def get_llm_with_structured_output(self, schema: dict | type, tier: ModelTier) -> BaseChatModel:
        if tier == ModelTier.LOCAL:
            llm = self._build_local()
        elif tier == ModelTier.REMOTE:
            llm = self._build_remote()

        return llm.with_structured_output(schema)
    

"""The two message types the bundled classifiers build their judge prompts with — the
definitions from the framework's `api_client/model_client.py`, without the provider clients."""
from dataclasses import dataclass
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: MessageRole
    content: str

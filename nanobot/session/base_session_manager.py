"""Base session management for conversation history."""

from dataclasses import dataclass
from typing import Any

from nanobot.utils.helpers import find_legal_message_start
from nanobot.session.manager import Session, SessionManager


@dataclass
class BaseSession(Session):


    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a legal tool-call boundary."""
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]

        # Avoid starting mid-turn when possible.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        # Drop orphan tool results at the front.
        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            if message["role"] == "tool":
                continue
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_call_id", "name", "reasoning_content"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out



class BaseSessionManager(SessionManager):

    def convert_session(self, session: Session) -> BaseSession:
        baseSession = BaseSession(
            session.key,
            session.messages,
            session.created_at,
            session.updated_at,
            session.metadata,
            session.last_consolidated
            )
        return baseSession

    def get_or_create(self, key: str) -> Session:
        session = super().get_or_create(key)
        return self.convert_session(session)

    def _load(self, key: str) -> Session | None:
        try:
            session = super()._load(key)
            return self.convert_session(session)
        except Exception as e:
            return None

    def save(self, session: Session) -> None:
        try:
            session = super().save(session)
            return self.convert_session(session)
        except Exception as e:
            return None
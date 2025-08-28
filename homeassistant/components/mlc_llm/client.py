"""MLC-LLM API client with OpenAI-compatible interface."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
import json
import logging
from typing import Any

import aiohttp
from pydantic import ValidationError

from homeassistant.const import CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DEFAULT_TIMEOUT
from .exception import MLCLLMError, MLCLLMRequestError, MLCLLMResponseError
from .openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    ChatTool,
)

_LOGGER = logging.getLogger(__name__)


class MLCLLMClient:
    """HTTP client for MLC-LLM API with OpenAI-compatible interface."""

    def __init__(
        self,
        base_url: str,
        *,
        session: aiohttp.ClientSession | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the client."""
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    @classmethod
    def create_from_config_entry(
        cls,
        hass: HomeAssistant,
        config_entry_data: dict,
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> MLCLLMClient:
        """Create client from config entry data."""
        session = async_get_clientsession(hass)
        return cls(
            base_url=config_entry_data[CONF_URL],
            session=session,
            timeout=timeout,
        )

    async def _request_raw(
        self,
        method: str,
        endpoint: str,
        *,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a raw HTTP request."""
        if self.session is None:
            raise MLCLLMRequestError("HTTP session is not initialized.")

        url = f"{self.base_url}{endpoint}"
        all_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if headers:
            all_headers.update(headers)

        try:
            async with self.session.request(
                method,
                url,
                json=json_data,
                headers=all_headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as err:
            error_text = await err.response.text() if err.response else str(err)
            raise MLCLLMResponseError(
                f"API error: {error_text}",
                status_code=err.status
            ) from err
        except aiohttp.ClientError as err:
            raise MLCLLMRequestError(f"Connection error: {err}") from err
        except TimeoutError as err:
            raise MLCLLMRequestError("Request timed out") from err
        except json.JSONDecodeError as err:
            raise MLCLLMResponseError(f"Invalid JSON response: {err}") from err

    async def chat_stream(
        self,
        *,
        messages: list[Mapping[str, Any] | ChatCompletionMessage],
        model: str | None = None, # in mlc-llm, a host only run a single model
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any] | ChatTool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Chat completion using OpenAI-compatible API with streaming."""

        if self.session is None:
            raise MLCLLMError("Session not initialized")

        try:
            chat_request = ChatCompletionRequest(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                **kwargs
            )
        except ValidationError as err:
            raise MLCLLMRequestError(f"Invalid request parameters: {err}") from err

        _LOGGER.debug("Sending streaming request to /v1/chat/completions: %s", {
            "model": chat_request.model,
            "messages_count": len(chat_request.messages),
            "temperature": chat_request.temperature,
            "max_tokens": chat_request.max_tokens,
        })

        json_data = chat_request.model_dump(exclude_none=True)

        async def _stream_response():
            try:
                async with self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=json_data,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    timeout=self.timeout,
                ) as response:
                        response.raise_for_status()

                        async for line in response.content:
                            line = line.decode("utf-8").strip()
                            if not line:
                                continue

                            # Handle Server-Sent Events format
                            if line.startswith("data: "):
                                line = line.removeprefix("data: ")  # Remove "data: " prefix

                            if line == "[DONE]":
                                _LOGGER.debug("Stream ended with [DONE] signal")
                                break

                            try:
                                chunk_data = json.loads(line)
                                if error := chunk_data.get("error"):
                                    raise MLCLLMResponseError(f"Stream error: {error}")

                                # Create and yield stream response
                                yield ChatCompletionStreamResponse(**chunk_data)

                            except json.JSONDecodeError:
                                _LOGGER.debug("Skipping invalid JSON line: %s", line)
                                continue
                            except ValidationError as err:
                                _LOGGER.debug("Skipping invalid chunk format: %s", err)
                                continue
            except aiohttp.ClientResponseError as err:
                error_text = await err.response.text() if err.response else str(err)
                raise MLCLLMResponseError(
                    f"API error: {error_text}",
                    status_code=err.status
                ) from err
            except aiohttp.ClientError as err:
                raise MLCLLMRequestError(f"Connection error: {err}") from err
            except TimeoutError as err:
                raise MLCLLMRequestError("Request timed out") from err

        return _stream_response()


    async def list_models(self) -> list[str]:
        """List available models."""
        try:
            response_json = await self._request_raw("GET", "/v1/models")
            return [model["id"] for model in response_json["data"]]

        except (KeyError, TypeError) as err:
            raise MLCLLMResponseError(f"Invalid models response format: {err}") from err

    async def close(self) -> None:
        """Close the client session if it was created by this client."""
        # Note: We don't close the session here because it's managed by Home Assistant
        # when created via async_get_clientsession()


    def __repr__(self) -> str:
        """Return string representation of the client."""
        return f"MLCLLMClient(host={self.host}, port={self.port})"

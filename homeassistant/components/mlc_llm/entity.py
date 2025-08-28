"""Base entity for the MLC-LLM integration."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
import json
import logging
from typing import Any

from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity

from . import MLCLLMConfigEntry
from .const import (
    CONF_MAX_HISTORY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_TEMPERATURE,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)
from .exception import MLCLLMError
from .openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionStreamResponse,
    ChatFunction,
    ChatFunctionCall,
    ChatTool,
    ChatToolCall,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

_LOGGER = logging.getLogger(__name__)

def _format_tool(
    tool: llm.Tool,
    custom_serializer: Callable[[Any], Any] | None,
) -> ChatTool:
    """Format tool specification."""
    tool_spec = ChatFunction(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatTool(type="function", function=tool_spec)

def _convert_content(
    content: conversation.Content,
) -> ChatCompletionMessage | None:
    """Convert any native chat message for this agent to the native format."""

    if isinstance(content, conversation.ToolResultContent):
        return ChatCompletionMessage(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json.dumps(content.tool_result),
        )
    if isinstance(content, conversation.AssistantContent):
        return ChatCompletionMessage(
            role="assistant",
            content=content.content,
            tool_calls=[
                ChatToolCall(
                    type="function",
                    id=tool_call.id,
                    function=ChatFunctionCall(
                        name=tool_call.tool_name,
                        arguments=tool_call.tool_args,
                    )
                )
                for tool_call in content.tool_calls or ()
            ],
        )
    if isinstance(content, conversation.UserContent):
        return ChatCompletionMessage(
            role="user",
            content=content.content,
        )
    if isinstance(content, conversation.SystemContent):
        return ChatCompletionMessage(
            role="system",
            content=content.content,
        )

    return None


def _decode_tool_arguments(arguments: str) -> Any:
    """Decode tool call arguments."""
    try:
        return json.loads(arguments)
    except json.JSONDecodeError as err:
        raise HomeAssistantError(f"Unexpected tool argument response: {err}") from err

async def _transform_stream(
    result: AsyncIterator[ChatCompletionStreamResponse],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the OpenRouter message to a ChatLog format."""

    async for chunk in result:
        choice = chunk.choices[0]
        data = conversation.AssistantContentDeltaDict(
            {
                "role": "assistant",
                "content": choice.delta.content,
                "tool_calls": [
                    llm.ToolInput(
                        {
                            "id": tool_call.id,
                            "tool_name": tool_call.function.name,
                            "tool_args": _decode_tool_arguments(
                                tool_call.function.arguments or "{}"
                            ),
                        }
                    )
                    for tool_call in choice.delta.tool_calls or ()
                ],
            }
        )
        yield data

class MLCLLMBaseEntity(Entity):
    """Base entity for MLC-LLM."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self,
        entry: MLCLLMConfigEntry,
        subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            manufacturer="MLC-LLM",
            name=subentry.title,
            entry_type=dr.DeviceEntryType.SERVICE,
            model=subentry.data.get(CONF_MODEL),
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
    ) -> None:
        """Generate an answer for the chat log."""
        settings = {**self.entry.data, **self.subentry.data}

        max_tokens = int(settings.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        temperature = settings.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        max_messages = int(settings.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))

        client = self.entry.runtime_data
        model = settings[CONF_MODEL]

        tools: list[ChatTool] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        message_history: list[ChatCompletionMessage] = [
            _convert_content(content) for content in chat_log.content
        ]
        self._trim_history(message_history, max_messages)

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                stream_generator = await client.chat_stream(
                    model=model,
                    messages=message_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                )

            except MLCLLMError as err:
                _LOGGER.error("Unexpected error talking to MLC-LLM server: %s", err)
                raise HomeAssistantError(
                    f"Sorry, I had a problem talking to the MLC-LLM server: {err}"
                ) from err

            message_history.extend(
                [
                    _convert_content(content)
                    async for content in chat_log.async_add_delta_content_stream(
                        self.entity_id, _transform_stream(stream_generator)
                    )
                ]
            )

            if not chat_log.unresponded_tool_results:
                break



    def _trim_history(
            self,
            message_history: list[ChatCompletionMessage],
            max_messages: int) -> None:
        """Trim the message history to the maximum allowed tokens."""
        if max_messages < 1:
            # Keep all messages
            return

        num_previous_rounds = sum(m.role == "user" for m in message_history) - 1
        if num_previous_rounds >= max_messages:
            # Remove the oldest messages, keeping the system prompt if any
            num_keep = 2 * max_messages + 1
            drop_index = len(message_history) - num_keep
            message_history = [
                message_history[0],
                *message_history[drop_index:],
            ]

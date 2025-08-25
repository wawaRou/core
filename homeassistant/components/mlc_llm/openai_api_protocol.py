"""Protocols in MLC LLM for OpenAI API."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
import shortuuid

from .const import DEFAULT_CHAT_COMPLETION_MAX_TOP_LOGPROBS
from .exception import MLCLLMRequestError


class ChatFunction(BaseModel):
    """Chat function definition."""
    description: str | None = None
    name: str
    parameters: dict


class ChatTool(BaseModel):
    """Chat tool definition."""
    type: Literal["function"]
    function: ChatFunction

class TopLogProbs(BaseModel):
    """Top log probabilities of a token."""
    token: str
    logprob: float
    bytes: list[int] | None


class LogProbsContent(BaseModel):
    """Log probabilities of a token."""
    token: str
    logprob: float
    bytes: list[int] | None
    top_logprobs: list[TopLogProbs] = []


class LogProbs(BaseModel):
    """Log probabilities of a completion."""
    content: list[LogProbsContent]


class CompletionLogProbs(BaseModel):
    """Completion log probabilities."""
    # The position of the token in the concatenated str: prompt + completion_text
    text_offset: list[int] | None
    token_logprobs: list[float]
    tokens: list[str]
    top_logprobs: list[dict[str, float]]

class CompletionUsage(BaseModel):
    """Completion usage."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    extra: dict[str, Any] | None = None
    """Extra metrics and info that may be returned by debug_config
    """

class ChatFunctionCall(BaseModel):
    """OpenAI chat function call protocol."""
    name: str
    arguments: dict[str, Any] | None = None


class ChatToolCall(BaseModel):
    """OpenAI chat tool call protocol."""
    id: str = Field(default_factory=lambda: f"call_{shortuuid.random()}")
    type: Literal["function"]
    function: ChatFunctionCall


class ChatCompletionMessage(BaseModel):
    """OpenAI chat completion message protocol."""
    content: str | list[dict] | None = None
    role: Literal["system", "user", "assistant", "tool"]
    name: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionStreamResponseChoice(BaseModel):
    """OpenAI completion stream response choice protocol."""
    finish_reason: Literal["stop", "length", "tool_calls", "error"] | None = None
    index: int = 0
    delta: ChatCompletionMessage
    logprobs: LogProbs | None = None

class ChatCompletionStreamResponse(BaseModel):
    """OpenAI completion stream response protocol."""
    id: str
    choices: list[ChatCompletionStreamResponseChoice]
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    system_fingerprint: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    usage: CompletionUsage | None = None

class RequestResponseFormat(BaseModel):
    """Request response format."""
    type: Literal["text", "json_object"] = "text"
    json_schema: str | None = Field(default=None, alias="schema")
    """This field is named json_schema instead of schema because BaseModel defines a method called
    schema. During construction of RequestResponseFormat, key "schema" still should be used:
    `RequestResponseFormat(type="json_object", schema="{}")`
    """

class StreamOptions(BaseModel):
    """Stream options for chat completion."""
    include_usage: bool | None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request protocol."""
    messages: list[ChatCompletionMessage]
    model: str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: dict[int, float] | None = None
    max_tokens: int | None = None
    n: int = 1
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ChatTool] | None = None
    tool_choice: Literal["none", "auto"] | dict | None = None
    user: str | None = None
    response_format: RequestResponseFormat | None = None

    @field_validator("frequency_penalty", "presence_penalty")
    @classmethod
    def check_penalty_range(cls, penalty_value: float | None) -> float | None:
        """Check if the penalty value is in range [-2, 2]."""
        if penalty_value and (penalty_value < -2 or penalty_value > 2):
            raise ValueError("Penalty value should be in range [-2, 2].")
        return penalty_value

    @field_validator("logit_bias")
    @classmethod
    def check_logit_bias(
        cls, logit_bias_value: dict[int, float] | None
    ) -> dict[int, float] | None:
        """Check if the logit bias key is given as an integer."""
        if logit_bias_value is None:
            return None
        for token_id, bias in logit_bias_value.items():
            if abs(bias) > 100:
                raise ValueError(
                    "Logit bias value should be in range [-100, 100], while value "
                    f"{bias} is given for token id {token_id}"
                )
        return logit_bias_value

    @model_validator(mode="after")
    def check_logprobs(self) -> "ChatCompletionRequest":
        """Check if the logprobs requirements are valid."""
        if self.top_logprobs < 0 or self.top_logprobs > DEFAULT_CHAT_COMPLETION_MAX_TOP_LOGPROBS:
            raise ValueError(
                f'"top_logprobs" must be in range [0, {DEFAULT_CHAT_COMPLETION_MAX_TOP_LOGPROBS}]'
            )
        if not self.logprobs and self.top_logprobs > 0:
            raise ValueError('"logprobs" must be True to support "top_logprobs"')
        return self

    @model_validator(mode="after")
    def check_stream_options(self) -> "ChatCompletionRequest":
        """Check stream options."""
        if self.stream_options is None:
            return self
        if not self.stream:
            raise ValueError("stream must be set to True when stream_options is present")
        return self

    def check_message_validity(self) -> None:
        """Check if the given chat messages are valid. Return error message if invalid."""
        for i, message in enumerate(self.messages):
            if message.role == "system" and i != 0:
                raise MLCLLMRequestError(
                    f"System prompt at position {i} in the message list is invalid."
                )
            if message.tool_call_id is not None:
                if message.role != "tool":
                    raise MLCLLMRequestError("Non-tool message having `tool_call_id` is invalid.")
            if isinstance(message.content, list):
                if message.role != "user":
                    raise MLCLLMRequestError("Non-user message having a list of content is invalid.")
            if message.tool_calls is not None:
                if message.role != "assistant":
                    raise MLCLLMRequestError("Non-assistant message having `tool_calls` is invalid.")
                raise MLCLLMRequestError("Assistant message having `tool_calls` is not supported yet.")


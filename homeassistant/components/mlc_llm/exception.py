"""Exceptions for MLC-LLM integration."""
from homeassistant.exceptions import HomeAssistantError


class MLCLLMError(HomeAssistantError):
    """Base exception for MLC-LLM integration."""

class MLCLLMRequestError(MLCLLMError):
    """Exception for request errors."""

class MLCLLMResponseError(MLCLLMError):
    """Exception for response errors."""
    def __init__(self, msg: str, status_code: int | None = None) -> None:
        """Initialize the exception."""
        super().__init__(msg)
        self.msg = msg
        self.status_code = status_code

    def __str__(self) -> str:
        """Str."""
        return f'{self.msg} (status code: {self.status_code})'

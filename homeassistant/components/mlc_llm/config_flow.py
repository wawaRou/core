"""Config flow for MLC-LLM integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME, CONF_URL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .client import MLCLLMClient
from .const import (
    CONF_MAX_HISTORY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DOMAIN,
)
from .exception import MLCLLMRequestError, MLCLLMResponseError

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
    }
)

class MLCLLMConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for MLC-LLM."""

    VERSION = 1
    MINOR_VERSION = 1

    def __init__(self) -> None:
        """Initialize config flow."""
        self.url: str | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}
        url = user_input[CONF_URL]

        # Check for duplicate entries
        self._async_abort_entries_match({CONF_URL: url})

        # Validate URL format
        try:
            url = cv.url(url)
        except vol.Invalid:
            errors["base"] = "invalid_url"
            return self.async_show_form(
                step_id="user",
                data_schema=self.add_suggested_values_to_schema(
                    STEP_USER_DATA_SCHEMA, user_input
                ),
                errors=errors,
            )

        # Test connection to MLC-LLM server
        try:
            session = async_get_clientsession(self.hass)
            client = MLCLLMClient(base_url=url, session=session)
            async with asyncio.timeout(DEFAULT_TIMEOUT):
                await client.list_models()
        except (MLCLLMRequestError, TimeoutError):
            errors["base"] = "cannot_connect"
        except MLCLLMResponseError:
            errors["base"] = "invalid_response"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"

        if errors:
            return self.async_show_form(
                step_id="user",
                data_schema=self.add_suggested_values_to_schema(
                    STEP_USER_DATA_SCHEMA, user_input
                ),
                errors=errors,
            )

        return self.async_create_entry(
            title=f"MLC-LLM ({url})",
            data={CONF_URL: url},
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": MLCLLMSubentryFlowHandler,
        }


class MLCLLMSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing MLC-LLM conversation subentries."""
    def __init__(self) -> None:
        """Initialize the subentry flow."""
        super().__init__()
        self._name: str | None = None
        self._available_models: list[str] = []

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle conversation agent configuration step."""
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is None:
            # Get the MLC-LLM client from runtime data
            entry = self._get_entry()
            client: MLCLLMClient = entry.runtime_data

            # Get available models from MLC-LLM server
            try:
                async with asyncio.timeout(DEFAULT_TIMEOUT):
                    self._available_models = await client.list_models()
                    if not self._available_models:
                        return self.async_abort(reason="no_models_available")
            except (MLCLLMRequestError, MLCLLMResponseError, TimeoutError):
                _LOGGER.exception("Failed to get models from MLC-LLM server")
                return self.async_abort(reason="cannot_connect")
            except Exception:
                _LOGGER.exception("Unexpected exception while getting models")
                return self.async_abort(reason="unknown")

            # Prepare form data
            if self._is_new:
                options = {}
                # Default name based on first available model
                default_name = f"MLC-LLM {self._available_models[0]}" if self._available_models else DEFAULT_CONVERSATION_NAME
            else:
                options = self._get_reconfigure_subentry().data.copy()
                default_name = None

            return self.async_show_form(
                step_id="set_options",
                data_schema=vol.Schema(
                    mlc_llm_config_option_schema(
                        self.hass,
                        self._is_new,
                        options,
                        self._available_models,
                        default_name,
                    )
                ),
            )

        # Process user input
        if self._is_new:
            self._name = user_input.pop(CONF_NAME)
            return self.async_create_entry(
                title=self._name,
                data=user_input,
            )

        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            data=user_input,
        )

    async_step_user = async_step_set_options
    async_step_reconfigure = async_step_set_options

def mlc_llm_config_option_schema(
    hass: HomeAssistant,
    is_new: bool,
    options: dict[str, Any],
    available_models: list[str],
    default_name: str | None = None,
) -> dict:
    """MLC-LLM conversation options schema."""
    schema: dict = {}
    if is_new and default_name:
        schema[vol.Required(CONF_NAME, default=default_name)] = str

    model_options = [
        SelectOptionDict(label=model, value=model) for model in available_models
    ]

    schema.update({
        vol.Required(
            CONF_MODEL,
            description={"suggested_value": options.get(CONF_MODEL, available_models[0] if available_models else "")},
        ): SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode="dropdown",
            )
        ),

        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),

        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
        ): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(
                        label=api.name,
                        value=api.id,
                    )
                    for api in llm.async_get_apis(hass)
                ],
                multiple=True,
            )
        ),

        vol.Optional(
            CONF_TEMPERATURE,
            description={
                "suggested_value": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
            },
        ): NumberSelector(
            NumberSelectorConfig(
                min=0.0,
                max=2.0,
                step=0.1,
                mode=NumberSelectorMode.BOX,
            )
        ),

        vol.Optional(
            CONF_MAX_TOKENS,
            description={
                "suggested_value": options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
            },
        ): NumberSelector(
            NumberSelectorConfig(
                min=1,
                max=32768,
                step=1,
                mode=NumberSelectorMode.BOX,
            )
        ),

        vol.Optional(
            CONF_MAX_HISTORY,
            description={
                "suggested_value": options.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)
            },
        ): NumberSelector(
            NumberSelectorConfig(
                min=0,
                max=100,
                step=1,
                mode=NumberSelectorMode.BOX,
            )
        ),
    })

    return schema


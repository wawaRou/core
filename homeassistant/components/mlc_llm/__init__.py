"""The MLC-LLM integration for Home Assistant."""

import asyncio

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .client import MLCLLMClient
from .const import DEFAULT_TIMEOUT
from .exception import MLCLLMError

# 支持的平台
PLATFORMS: list[Platform] = [Platform.CONVERSATION]

type MLCLLMConfigEntry = ConfigEntry[MLCLLMClient]


async def async_setup_entry(hass: HomeAssistant, entry: MLCLLMConfigEntry) -> bool:
    """Set up MLC-LLM from a config entry."""
    settings = {**entry.data, **entry.options}
    client = MLCLLMClient(
        base_url=settings[CONF_URL],
        session=async_get_clientsession(hass)
    )
    try:
        async with asyncio.timeout(DEFAULT_TIMEOUT):
            await client.list_models()
    except (TimeoutError, MLCLLMError) as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: MLCLLMConfigEntry) -> bool:
    """Unload MLC-LLM."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    return True


async def async_update_options(hass: HomeAssistant, entry: MLCLLMConfigEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)

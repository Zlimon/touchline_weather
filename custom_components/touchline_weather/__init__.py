"""The Touchline Weather integration."""
from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Touchline Weather component."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry) -> bool:
    """Set up Touchline Weather from a config entry."""
    return True


async def async_unload_entry(hass: HomeAssistant, entry) -> bool:
    """Unload a config entry."""
    return True
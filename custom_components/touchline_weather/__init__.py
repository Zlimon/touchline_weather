"""Touchline Weather Adaptive Controller integration."""
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Touchline Weather component."""
    return True

async def async_setup_entry(hass: HomeAssistant, entry) -> bool:
    """Set up from a config entry."""
    # This would be for UI configuration, which you're not using yet
    return True
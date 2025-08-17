"""The Touchline Weather integration."""
from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType
from homeassistant.const import Platform

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.CLIMATE]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Touchline Weather component."""
    try:
        _LOGGER.info("Setting up Touchline Weather integration")

        # Initialize the domain data
        hass.data.setdefault(DOMAIN, {})

        return True
    except Exception as err:
        _LOGGER.error(f"Error setting up Touchline Weather integration: {err}")
        return False


async def async_setup_entry(hass: HomeAssistant, entry) -> bool:
    """Set up Touchline Weather from a config entry."""
    try:
        _LOGGER.info("Setting up Touchline Weather config entry")
        return True
    except Exception as err:
        _LOGGER.error(f"Error setting up Touchline Weather config entry: {err}")
        return False


async def async_unload_entry(hass: HomeAssistant, entry) -> bool:
    """Unload a config entry."""
    try:
        _LOGGER.info("Unloading Touchline Weather config entry")
        return True
    except Exception as err:
        _LOGGER.error(f"Error unloading Touchline Weather config entry: {err}")
        return False
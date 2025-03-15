"""Platform for Roth Touchline floor heating controller with weather adaptation."""

from __future__ import annotations

from typing import Any, NamedTuple
import datetime
import aiohttp
import asyncio
import logging

from pytouchline_extended import PyTouchline
import voluptuous as vol

from homeassistant.components.climate import (
    PLATFORM_SCHEMA as CLIMATE_PLATFORM_SCHEMA,
    ClimateEntity,
    ClimateEntityFeature,
    HVACMode,
)
from homeassistant.const import (
    ATTR_TEMPERATURE,
    CONF_HOST,
    CONF_API_KEY,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    CONF_WEATHER_API_KEY,
    CONF_LATITUDE,
    CONF_LONGITUDE,
    CONF_WEATHER_ADAPTATION,
    CONF_BASE_TEMP,
    CONF_ADJUSTMENT_FACTOR,
    CONF_UPDATE_INTERVAL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

# Default values
DEFAULT_UPDATE_INTERVAL = datetime.timedelta(hours=1)
DEFAULT_BASE_TEMP = 20.0  # Base temperature in Celsius
DEFAULT_ADJUSTMENT_FACTOR = 0.5  # How aggressively to adjust (higher = more aggressive)


class PresetMode(NamedTuple):
    """Settings for preset mode."""

    mode: int
    program: int


PRESET_MODES = {
    "Normal": PresetMode(mode=0, program=0),
    "Night": PresetMode(mode=1, program=0),
    "Holiday": PresetMode(mode=2, program=0),
    "Pro 1": PresetMode(mode=0, program=1),
    "Pro 2": PresetMode(mode=0, program=2),
    "Pro 3": PresetMode(mode=0, program=3),
    "Weather Adaptive": PresetMode(mode=0, program=0),  # Added weather adaptive mode
}

TOUCHLINE_HA_PRESETS = {
    (settings.mode, settings.program): preset
    for preset, settings in PRESET_MODES.items()
}

# Extended platform schema with weather API configuration
PLATFORM_SCHEMA = CLIMATE_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_WEATHER_API_KEY): cv.string,
    vol.Optional(CONF_LATITUDE): cv.latitude,
    vol.Optional(CONF_LONGITUDE): cv.longitude,
    vol.Optional(CONF_WEATHER_ADAPTATION, default=False): cv.boolean,
    vol.Optional(CONF_BASE_TEMP, default=DEFAULT_BASE_TEMP): vol.Coerce(float),
    vol.Optional(CONF_ADJUSTMENT_FACTOR, default=DEFAULT_ADJUSTMENT_FACTOR): vol.Coerce(float),
    vol.Optional(CONF_UPDATE_INTERVAL, default=DEFAULT_UPDATE_INTERVAL): vol.All(
        cv.time_period, cv.positive_timedelta
    ),
})


class TouchlineDataUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage fetching Touchline data."""

    def __init__(
        self,
        hass: HomeAssistant,
        logger: logging.Logger,
        py_touchline: PyTouchline,
        update_interval: datetime.timedelta,
    ):
        """Initialize the coordinator."""
        super().__init__(
            hass,
            logger,
            name=DOMAIN,
            update_interval=update_interval,
        )
        self.py_touchline = py_touchline
        self.device_count = None
        self.devices = {}

    async def _async_update_data(self):
        """Fetch data from API endpoint."""
        try:
            # Initial fetch to get number of devices (if not already done)
            if self.device_count is None:
                self.device_count = await self.hass.async_add_executor_job(
                    self.py_touchline.get_number_of_devices
                )
                self.device_count = int(self.device_count)
                _LOGGER.debug(f"Found {self.device_count} Touchline devices")

                # Initialize devices
                for device_id in range(self.device_count):
                    self.devices[device_id] = PyTouchline(id=device_id, url=self.py_touchline._url)

            # Update all devices
            for device_id, device in self.devices.items():
                await self.hass.async_add_executor_job(device.update)

            return self.devices
        except Exception as err:
            _LOGGER.error(f"Error communicating with Touchline: {err}")
            raise


class WeatherManager:
    """Manage weather data and temperature calculations."""

    def __init__(self, hass, api_key, latitude, longitude, base_temp, adjustment_factor):
        """Initialize the weather manager."""
        self.hass = hass
        self.api_key = api_key
        self.latitude = latitude
        self.longitude = longitude
        self.base_temp = base_temp
        self.adjustment_factor = adjustment_factor
        self.forecast_data = None
        self.avg_forecast_temp = None
        self._callback_listeners = []

    async def update_forecast(self, *_):
        """Update the weather forecast data."""
        try:
            # Example using OpenWeatherMap API - you'll need to adapt this to your specific weather API
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={self.latitude}&lon={self.longitude}&appid={self.api_key}&units=metric"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        _LOGGER.error(f"Failed to get weather data: {response.status}")
                        return

                    data = await response.json()
                    self.forecast_data = data

                    # Calculate average temperature for next 6 hours (assuming hourly forecasts)
                    forecast_items = data.get('list', [])[:6]  # Get first 6 items
                    if forecast_items:
                        temps = [item.get('main', {}).get('temp', 0) for item in forecast_items]
                        self.avg_forecast_temp = sum(temps) / len(temps)
                        _LOGGER.info(f"Average forecast temperature for next 6 hours: {self.avg_forecast_temp}°C")

                        # Notify all registered devices about the update
                        for callback in self._callback_listeners:
                            callback()

        except Exception as err:
            _LOGGER.error(f"Error updating weather forecast: {err}")

    def calculate_target_temperature(self, current_target):
        """Calculate the adjusted target temperature based on the forecast."""
        if self.avg_forecast_temp is None:
            return current_target

        # Simple algorithm: adjust target temperature based on forecast deviation from base temperature
        # If forecast is colder than base temp, increase target temp (and vice versa)
        temp_difference = self.base_temp - self.avg_forecast_temp
        adjustment = temp_difference * self.adjustment_factor

        new_target = current_target + adjustment

        # Reasonable limits
        new_target = max(min(new_target, 28), 16)

        _LOGGER.info(
            f"Weather-adaptive adjustment: forecast avg={self.avg_forecast_temp}°C, "
            f"base={self.base_temp}°C, adjustment={adjustment:.1f}°C, "
            f"new target={new_target:.1f}°C"
        )

        return round(new_target, 1)

    def register_callback(self, callback):
        """Register a callback for weather updates."""
        if callback not in self._callback_listeners:
            self._callback_listeners.append(callback)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Touchline devices with weather adaptation."""

    host = config[CONF_HOST]
    weather_adaptation = config[CONF_WEATHER_ADAPTATION]

    # Weather API settings
    weather_api_key = config.get(CONF_WEATHER_API_KEY)
    latitude = config.get(CONF_LATITUDE, hass.config.latitude)
    longitude = config.get(CONF_LONGITUDE, hass.config.longitude)
    base_temp = config[CONF_BASE_TEMP]
    adjustment_factor = config[CONF_ADJUSTMENT_FACTOR]
    update_interval = config[CONF_UPDATE_INTERVAL]

    # Check if weather adaptation is enabled but missing API key
    if weather_adaptation and not weather_api_key:
        _LOGGER.error("Weather adaptation enabled but no API key provided")
        return

    # Create base PyTouchline instance
    py_touchline = PyTouchline(url=host)

    # Create data update coordinator
    coordinator = TouchlineDataUpdateCoordinator(
        hass,
        _LOGGER,
        py_touchline,
        update_interval,
    )

    # Initial data fetch
    await coordinator.async_config_entry_first_refresh()

    # Setup weather manager if adaptation is enabled
    weather_manager = None
    if weather_adaptation:
        weather_manager = WeatherManager(
            hass,
            weather_api_key,
            latitude,
            longitude,
            base_temp,
            adjustment_factor
        )
        await weather_manager.update_forecast()

    # Create device entities
    devices = []
    for device_id in range(coordinator.device_count):
        device = WeatherAdaptiveTouchline(
            coordinator,
            device_id,
            weather_manager,
            weather_adaptation
        )
        devices.append(device)

    async_add_entities(devices, False)  # False because we already updated via coordinator

    # Set up periodic weather updates if adaptation is enabled
    if weather_adaptation:
        async_track_time_interval(
            hass, weather_manager.update_forecast, update_interval
        )


class WeatherAdaptiveTouchline(ClimateEntity):
    """Representation of a Touchline device with weather adaptation."""

    _attr_hvac_mode = HVACMode.HEAT
    _attr_hvac_modes = [HVACMode.HEAT]
    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE
    )
    _attr_temperature_unit = UnitOfTemperature.CELSIUS

    def __init__(self, coordinator, device_id, weather_manager=None, weather_adaptation=False):
        """Initialize the Touchline device."""
        self.coordinator = coordinator
        self.device_id = device_id
        self._weather_manager = weather_manager
        self._weather_adaptation = weather_adaptation
        self._weather_adaptive_mode = False
        self._name = None
        self._attr_unique_id = f"touchline_weather_{device_id}"

        # Register for weather updates if weather adaptation is enabled
        if self._weather_manager:
            self._weather_manager.register_callback(self.weather_update_callback)

    def weather_update_callback(self):
        """Handle weather forecast updates."""
        if self._weather_adaptive_mode:
            self.update_weather_adaptive_temperature()
            self.async_write_ha_state()

    async def update_weather_adaptive_temperature(self):
        """Update target temperature based on weather forecast."""
        if not self._weather_manager or not self._weather_adaptive_mode:
            return

        current_target = self.target_temperature
        new_target = self._weather_manager.calculate_target_temperature(current_target)

        if abs(new_target - current_target) >= 0.2:  # Only update if significant change
            _LOGGER.info(f"Setting new weather-adaptive temperature for {self.name}: {new_target}°C")
            await self.hass.async_add_executor_job(
                self.coordinator.devices[self.device_id].set_target_temperature, new_target
            )
            await self.coordinator.async_request_refresh()

    @property
    def device_info(self):
        """Return device information about this entity."""
        return {
            "identifiers": {(DOMAIN, f"touchline_{self.device_id}")},
            "name": self.name,
            "manufacturer": "Roth",
            "model": "Touchline",
        }

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self.coordinator.last_update_success

    @property
    def name(self):
        """Return the name of the climate device."""
        device = self.coordinator.devices[self.device_id]
        return device.get_name()

    @property
    def current_temperature(self):
        """Return the current temperature."""
        device = self.coordinator.devices[self.device_id]
        return device.get_current_temperature()

    @property
    def target_temperature(self):
        """Return the temperature we try to reach."""
        device = self.coordinator.devices[self.device_id]
        return device.get_target_temperature()

    @property
    def preset_mode(self):
        """Return the current preset mode."""
        if self._weather_adaptive_mode:
            return "Weather Adaptive"

        device = self.coordinator.devices[self.device_id]
        operation_mode = device.get_operation_mode()
        week_program = device.get_week_program()
        return TOUCHLINE_HA_PRESETS.get((operation_mode, week_program))

    @property
    def preset_modes(self):
        """Return available preset modes."""
        modes = list(PRESET_MODES)
        if not self._weather_adaptation:
            modes.remove("Weather Adaptive")
        return modes

    async def async_set_preset_mode(self, preset_mode):
        """Set new target preset mode."""
        self._weather_adaptive_mode = preset_mode == "Weather Adaptive"

        if not self._weather_adaptive_mode:
            # Set the standard preset mode
            mode_settings = PRESET_MODES[preset_mode]
            device = self.coordinator.devices[self.device_id]
            await self.hass.async_add_executor_job(device.set_operation_mode, mode_settings.mode)
            await self.hass.async_add_executor_job(device.set_week_program, mode_settings.program)
        else:
            # If switching to weather adaptive mode, immediately apply temperature adjustment
            if self._weather_manager:
                await self.update_weather_adaptive_temperature()

        # Request refresh to update state
        await self.coordinator.async_request_refresh()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set new target hvac mode."""
        self._current_operation_mode = HVACMode.HEAT

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if kwargs.get(ATTR_TEMPERATURE) is not None:
            target_temperature = kwargs.get(ATTR_TEMPERATURE)
            # Turn off weather adaptive mode if user manually sets temperature
            self._weather_adaptive_mode = False
            # Use the coordinator's device instead of self.unit
            await self.hass.async_add_executor_job(
                self.coordinator.devices[self.device_id].set_target_temperature,
                target_temperature
            )
            # Request refresh to update state
            await self.coordinator.async_request_refresh()
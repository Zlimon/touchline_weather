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
    CONF_DEFAULT_ADAPTIVE_DEVICES,
    CONF_NAME_TEMPLATE,
    CONF_NAMES,
    CONF_COMFORT_TEMPS,
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
    vol.Optional(CONF_DEFAULT_ADAPTIVE_DEVICES, default=[]): vol.All(cv.ensure_list, [cv.positive_int]),
    vol.Optional(CONF_NAME_TEMPLATE, default="Touchline {id}"): cv.string,
    vol.Optional(CONF_NAMES, default={}): {cv.positive_int: cv.string},
    vol.Optional(CONF_COMFORT_TEMPS, default={}): {cv.positive_int: vol.Coerce(float)},
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
        self.last_update_time = None  # Track when we last updated

    async def update_forecast(self, *_):
        """Update the weather forecast data using Yr API."""
        # Prevent duplicate updates within 30 seconds
        now = datetime.datetime.now()
        if self.last_update_time and (now - self.last_update_time).total_seconds() < 30:
            _LOGGER.debug(
                f"Skipping duplicate update, last update was {(now - self.last_update_time).total_seconds()} seconds ago")
            return

        self.last_update_time = now
        _LOGGER.info(f"Weather update triggered at {now}")

        try:
            # Yr API endpoint for location forecast
            url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={self.latitude}&lon={self.longitude}"

            headers = {
                "User-Agent": "homeassistant-touchline/1.0 github.com/Zlimon/touchline_weather",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        _LOGGER.error(f"Failed to get weather data from Yr: {response.status}, {await response.text()}")
                        return

                    data = await response.json()
                    self.forecast_data = data

                    # Extract temperature data for next 6 hours
                    forecast_items = data.get('properties', {}).get('timeseries', [])[:6]  # Get first 6 items
                    if forecast_items:
                        temps = [
                            item.get('data', {}).get('instant', {}).get('details', {}).get('air_temperature', 0)
                            for item in forecast_items
                        ]
                        self.avg_forecast_temp = sum(temps) / len(temps)
                        _LOGGER.info(f"Average forecast temperature for next 6 hours: {self.avg_forecast_temp}°C")

                        # Only notify callbacks once
                        await self.notify_listeners()

        except Exception as err:
            _LOGGER.error(f"Error updating weather forecast from Yr: {err}")

    async def notify_listeners(self):
        """Notify all registered callbacks about weather update."""
        _LOGGER.debug(f"Notifying {len(self._callback_listeners)} listeners of weather update")
        # Create tasks for each callback
        for callback in self._callback_listeners:
            if asyncio.iscoroutinefunction(callback):
                await callback()  # Await directly for more predictable behavior
            else:
                callback()

    def calculate_target_temperature(self, base_comfort_temp, current_room_temp=None):
        """Calculate the adjusted target temperature based on the forecast and current room temperature.

        Args:
            base_comfort_temp: The baseline comfortable temperature without weather adjustment
            current_room_temp: The current measured room temperature (if available)
        """
        if self.avg_forecast_temp is None:
            return base_comfort_temp

        # Basic weather adjustment as before
        temp_difference = self.base_temp - self.avg_forecast_temp
        weather_adjustment = temp_difference * self.adjustment_factor

        # Start with the weather-adjusted target
        new_target = base_comfort_temp + weather_adjustment

        # If we have current room temperature, factor it in
        if current_room_temp is not None:
            # If room is already warmer than comfort temp, reduce the adjustment
            temp_surplus = current_room_temp - base_comfort_temp
            if temp_surplus > 0:
                # Reduce target by a portion of the surplus
                # The 0.7 factor prevents oscillation (can be tuned)
                reduction = temp_surplus * 0.7
                new_target = new_target - reduction
                _LOGGER.info(
                    f"Room temperature adjustment: current={current_room_temp}°C, "
                    f"surplus={temp_surplus:.1f}°C, reduction={reduction:.1f}°C"
                )

        # Reasonable limits
        new_target = max(min(new_target, 28), 16)

        _LOGGER.info(
            f"Weather-adaptive adjustment: forecast avg={self.avg_forecast_temp}°C, "
            f"base temp={self.base_temp}°C, base comfort={base_comfort_temp}°C, "
            f"adjustment={weather_adjustment:.1f}°C, new target={new_target:.1f}°C"
        )

        return round(new_target * 2) / 2

    def register_callback(self, callback):
        """Register a callback for weather updates."""
        if callback not in self._callback_listeners:
            self._callback_listeners.append(callback)
            _LOGGER.debug(f"Registered new callback, total callbacks: {len(self._callback_listeners)}")


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Touchline devices with weather adaptation."""

    host = config[CONF_HOST]
    weather_adaptation = config[CONF_WEATHER_ADAPTATION]
    default_adaptive_devices = config.get(CONF_DEFAULT_ADAPTIVE_DEVICES, [])

    # Weather API settings
    weather_api_key = config.get(CONF_WEATHER_API_KEY)
    latitude = config.get(CONF_LATITUDE, hass.config.latitude)
    longitude = config.get(CONF_LONGITUDE, hass.config.longitude)
    base_temp = config[CONF_BASE_TEMP]
    adjustment_factor = config[CONF_ADJUSTMENT_FACTOR]
    update_interval = config[CONF_UPDATE_INTERVAL]

    name_template = config.get(CONF_NAME_TEMPLATE, "Touchline {id}")
    names = config.get(CONF_NAMES, {})

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

    comfort_temps = config.get(CONF_COMFORT_TEMPS, {})

    # Create device entities
    devices = []
    for device_id in range(coordinator.device_count):
        device_id_plus_one = device_id + 1
        comfort_temp = comfort_temps.get(device_id_plus_one, 22.0)  # Default to 22°C
        default_adaptive = device_id_plus_one in default_adaptive_devices

        device = WeatherAdaptiveTouchline(
            coordinator,
            device_id,
            weather_manager,
            weather_adaptation,
            default_adaptive,
            name_template,
            names,
            comfort_temp
        )
        devices.append(device)

    async_add_entities(devices, False)  # False because we already updated via coordinator

    # Register a service to manually trigger weather updates
    if weather_adaptation:
        async def handle_force_weather_update(call):
            """Handle the service call to force a weather update."""
            _LOGGER.info("Manual weather update triggered")
            await weather_manager.update_forecast()

        # Register our new service
        hass.services.async_register(
            DOMAIN, 'force_weather_update', handle_force_weather_update
        )

    if weather_adaptation:
        _LOGGER.info(f"Setting up scheduled updates every {update_interval}")

        # Initial update
        await weather_manager.update_forecast()

        # Schedule future updates
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

    def __init__(self, coordinator, device_id, weather_manager=None, weather_adaptation=False,
                 default_adaptive=False, name_template=None, names=None, comfort_temp=22.0):
        """Initialize the Touchline device."""
        self.coordinator = coordinator
        self.device_id = device_id
        self._weather_manager = weather_manager
        self._weather_adaptation = weather_adaptation
        self._weather_adaptive_mode = default_adaptive
        self._name_template = name_template or "Touchline {id}"
        self._names = names or {}
        self._base_comfort_temp = comfort_temp
        self._attr_unique_id = f"touchline_weather_{device_id}"

        # Register for weather updates if weather adaptation is enabled
        if self._weather_manager:
            self._weather_manager.register_callback(self.weather_update_callback)

    async def weather_update_callback(self):
        """Handle weather forecast updates."""
        _LOGGER.debug(f"Weather update callback called for {self.name}, adaptive mode: {self._weather_adaptive_mode}")
        if self._weather_adaptive_mode:
            await self.update_weather_adaptive_temperature()
            self.async_write_ha_state()

    async def update_weather_adaptive_temperature(self):
        """Update target temperature based on weather forecast."""
        if not self._weather_manager or not self._weather_adaptive_mode:
            return

        # Get current room temperature
        current_room_temp = self.current_temperature

        # Use the stored base comfort temperature and current room temp
        new_target = self._weather_manager.calculate_target_temperature(
            self._base_comfort_temp,
            current_room_temp
        )

        # Get current target from device for logging purposes only
        current_target = self.target_temperature

        if abs(new_target - current_target) >= 0.2:  # Only update if significant change
            _LOGGER.info(f"Setting new weather-adaptive temperature for {self.name}: {new_target}°C")
            await self.hass.async_add_executor_job(
                self.coordinator.devices[self.device_id].set_target_temperature, new_target
            )
            await self.coordinator.async_request_refresh()

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

            if self._weather_adaptive_mode:
                # Store the new base comfort temperature
                self._base_comfort_temp = target_temperature
                _LOGGER.info(f"Updated base comfort temperature for {self.name} to {self._base_comfort_temp}°C")
                # Let weather adaptation adjust from this new base immediately
                await self.update_weather_adaptive_temperature()
            else:
                # Use the coordinator's device instead of self.unit
                await self.hass.async_add_executor_job(
                    self.coordinator.devices[self.device_id].set_target_temperature,
                    target_temperature
                )

            # Request refresh to update state
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
        # First check if there's a specific name for this device ID
        device_id_plus_one = self.device_id + 1  # Convert to 1-indexed for user convenience
        if device_id_plus_one in self._names:
            return self._names[device_id_plus_one]

        # Otherwise use the template with the ID
        return self._name_template.format(id=device_id_plus_one)

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
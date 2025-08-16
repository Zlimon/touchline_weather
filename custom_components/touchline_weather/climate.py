"""Platform for Roth Touchline floor heating controller with Sonoff sensor integration and weather adaptation."""

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
    CONF_SONOFF_SENSORS,  # NEW: Sonoff sensor mapping
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
    "Auto Weather": PresetMode(mode=0, program=0),  # Renamed for clarity
}

TOUCHLINE_HA_PRESETS = {
    (settings.mode, settings.program): preset
    for preset, settings in PRESET_MODES.items()
}

# Extended platform schema with Sonoff sensor configuration
PLATFORM_SCHEMA = CLIMATE_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_WEATHER_API_KEY): cv.string,
    vol.Optional(CONF_LATITUDE): cv.latitude,
    vol.Optional(CONF_LONGITUDE): cv.longitude,
    vol.Optional(CONF_WEATHER_ADAPTATION, default=True): cv.boolean,  # Default to True
    vol.Optional(CONF_BASE_TEMP, default=DEFAULT_BASE_TEMP): vol.Coerce(float),
    vol.Optional(CONF_ADJUSTMENT_FACTOR, default=DEFAULT_ADJUSTMENT_FACTOR): vol.Coerce(float),
    vol.Optional(CONF_UPDATE_INTERVAL, default=DEFAULT_UPDATE_INTERVAL): vol.All(
        cv.time_period, cv.positive_timedelta
    ),
    vol.Optional(CONF_DEFAULT_ADAPTIVE_DEVICES, default=[]): vol.All(cv.ensure_list, [cv.positive_int]),
    vol.Optional(CONF_NAME_TEMPLATE, default="Auto Climate {id}"): cv.string,
    vol.Optional(CONF_NAMES, default={}): {cv.positive_int: cv.string},
    vol.Optional(CONF_COMFORT_TEMPS, default={}): {cv.positive_int: vol.Coerce(float)},
    vol.Optional(CONF_SONOFF_SENSORS, default={}): {cv.positive_int: cv.string},  # NEW: Zone to sensor mapping
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
        self.forecast_temp_3h = None  # CHANGED: Store 3-hour forecast instead of average
        self._callback_listeners = []
        self.last_update_time = None

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

                    # CHANGED: Get temperature exactly 3 hours from now instead of averaging
                    forecast_items = data.get('properties', {}).get('timeseries', [])
                    target_time = now + datetime.timedelta(hours=3)
                    
                    # Find the forecast closest to 3 hours from now
                    closest_forecast = None
                    min_time_diff = float('inf')
                    
                    for item in forecast_items:
                        forecast_time_str = item.get('time')
                        if forecast_time_str:
                            # Parse the forecast time
                            forecast_time = datetime.datetime.fromisoformat(forecast_time_str.replace('Z', '+00:00'))
                            time_diff = abs((forecast_time - target_time.replace(tzinfo=forecast_time.tzinfo)).total_seconds())
                            
                            if time_diff < min_time_diff:
                                min_time_diff = time_diff
                                closest_forecast = item

                    if closest_forecast:
                        self.forecast_temp_3h = closest_forecast.get('data', {}).get('instant', {}).get('details', {}).get('air_temperature', 0)
                        _LOGGER.info(f"Temperature forecast for 3 hours from now: {self.forecast_temp_3h}°C")
                        await self.notify_listeners()
                    else:
                        _LOGGER.warning("Could not find suitable 3-hour forecast data")

        except Exception as err:
            _LOGGER.error(f"Error updating weather forecast from Yr: {err}")

    async def notify_listeners(self):
        """Notify all registered callbacks about weather update."""
        _LOGGER.debug(f"Notifying {len(self._callback_listeners)} listeners of weather update")
        for callback in self._callback_listeners:
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()

    def calculate_target_temperature(self, base_comfort_temp, current_room_temp=None):
        """Calculate the adjusted target temperature based on the 3-hour forecast and current conditions.

        Args:
            base_comfort_temp: The baseline comfortable temperature without weather adjustment
            current_room_temp: The current measured room temperature (if available)
        """
        if self.forecast_temp_3h is None:
            _LOGGER.warning("No 3-hour forecast available, using base comfort temperature")
            return base_comfort_temp

        # Check current time for different modes
        current_time = datetime.datetime.now().time()
        night_mode_start = datetime.time(0, 0)  # 00:00
        night_mode_end = datetime.time(4, 0)  # 04:00

        # Night mode (reduced heating)
        if night_mode_start <= current_time <= night_mode_end:
            night_temp = 18.0
            _LOGGER.info(f"Night mode active (00:00-04:00): Setting reduced target temperature of {night_temp}°C")
            return night_temp

        # CHANGED: Use 3-hour forecast instead of average
        temp_difference = self.base_temp - self.forecast_temp_3h
        weather_adjustment = temp_difference * self.adjustment_factor

        # Start with the weather-adjusted target
        new_target = base_comfort_temp + weather_adjustment

        # If we have current room temperature, factor it in for more intelligent control
        if current_room_temp is not None:
            # If room is significantly warmer than comfort temp, reduce heating demand
            temp_surplus = current_room_temp - base_comfort_temp
            if temp_surplus > 0.5:  # More than 0.5°C above comfort
                # Reduce target by a portion of the surplus
                reduction = temp_surplus * 0.8
                new_target = new_target - reduction
                _LOGGER.info(f"Room surplus detected: {temp_surplus:.1f}°C, reducing target by {reduction:.1f}°C")

            # If room is much colder, increase heating demand
            elif temp_surplus < -1.0:  # More than 1°C below comfort
                boost = abs(temp_surplus) * 0.3
                new_target = new_target + boost
                _LOGGER.info(f"Room deficit detected: {temp_surplus:.1f}°C, boosting target by {boost:.1f}°C")

        # Reasonable limits
        new_target = max(min(new_target, 28), 16)

        _LOGGER.info(
            f"Auto temperature calculation: 3h forecast={self.forecast_temp_3h}°C, "
            f"base temp={self.base_temp}°C, base comfort={base_comfort_temp}°C, "
            f"weather adjustment={weather_adjustment:.1f}°C, current room={current_room_temp}°C, "
            f"final target={new_target:.1f}°C"
        )

        return round(new_target * 2) / 2

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
    """Set up the Touchline devices with Sonoff integration and weather adaptation."""

    host = config[CONF_HOST]
    weather_adaptation = config[CONF_WEATHER_ADAPTATION]
    default_adaptive_devices = config.get(CONF_DEFAULT_ADAPTIVE_DEVICES, [])
    sonoff_sensors = config.get(CONF_SONOFF_SENSORS, {})  # NEW: Sonoff sensor mapping

    # Weather API settings
    weather_api_key = config.get(CONF_WEATHER_API_KEY)
    latitude = config.get(CONF_LATITUDE, hass.config.latitude)
    longitude = config.get(CONF_LONGITUDE, hass.config.longitude)
    base_temp = config[CONF_BASE_TEMP]
    adjustment_factor = config[CONF_ADJUSTMENT_FACTOR]
    update_interval = config[CONF_UPDATE_INTERVAL]

    name_template = config.get(CONF_NAME_TEMPLATE, "Auto Climate {id}")
    names = config.get(CONF_NAMES, {})

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

    # Setup weather manager (always enabled now for auto control)
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
        comfort_temp = comfort_temps.get(device_id_plus_one, 22.0)
        sonoff_sensor = sonoff_sensors.get(device_id_plus_one)  # NEW: Get Sonoff sensor for this zone
        default_adaptive = device_id_plus_one in default_adaptive_devices

        device = SonoffIntegratedTouchline(  # RENAMED class
            coordinator,
            device_id,
            weather_manager,
            weather_adaptation,
            default_adaptive,
            name_template,
            names,
            comfort_temp,
            sonoff_sensor  # NEW: Pass Sonoff sensor
        )
        devices.append(device)

    async_add_entities(devices, False)

    # Register services
    async def handle_force_weather_update(call):
        """Handle the service call to force a weather update."""
        _LOGGER.info("Manual weather update triggered")
        await weather_manager.update_forecast()

    hass.services.async_register(
        DOMAIN, 'force_weather_update', handle_force_weather_update
    )

    _LOGGER.info(f"Setting up scheduled updates every {update_interval}")

    # Initial update and schedule future updates
    await weather_manager.update_forecast()
    async_track_time_interval(
        hass, weather_manager.update_forecast, update_interval
    )


class SonoffIntegratedTouchline(ClimateEntity):  # RENAMED class
    """Representation of a Touchline device with Sonoff sensor integration and automatic weather adaptation."""

    _attr_hvac_mode = HVACMode.HEAT
    _attr_hvac_modes = [HVACMode.HEAT]
    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE
    )
    _attr_temperature_unit = UnitOfTemperature.CELSIUS

    def __init__(self, coordinator, device_id, weather_manager=None, weather_adaptation=False,
                 default_adaptive=False, name_template=None, names=None, comfort_temp=22.0, 
                 sonoff_sensor=None):  # NEW: Added sonoff_sensor parameter
        """Initialize the Touchline device with Sonoff integration."""
        self.coordinator = coordinator
        self.device_id = device_id
        self._weather_manager = weather_manager
        self._weather_adaptation = weather_adaptation
        self._auto_mode = default_adaptive  # CHANGED: Always start in auto mode
        self._name_template = name_template or "Auto Climate {id}"
        self._names = names or {}
        self._base_comfort_temp = comfort_temp
        self._sonoff_sensor = sonoff_sensor  # NEW: Store Sonoff sensor entity ID
        self._attr_unique_id = f"touchline_auto_{device_id}"
        self._startup_complete = False  # Track if startup is complete

        # Register for weather updates
        if self._weather_manager:
            self._weather_manager.register_callback(self.weather_update_callback)

    async def start_automatic_control(self):
        """Start automatic temperature control based on weather and Sonoff sensors."""
        if self._auto_mode and self._weather_manager and self._startup_complete:
            _LOGGER.info(f"Starting automatic control for {self.name}")
            await self.update_automatic_temperature()

    async def async_added_to_hass(self) -> None:
        """Called when entity is added to Home Assistant."""
        await super().async_added_to_hass()
        
        # Now we can safely start automatic control
        if self._auto_mode and self._weather_manager:
            await self.start_automatic_control()
        
        self._startup_complete = True

    async def weather_update_callback(self):
        """Handle weather forecast updates - trigger automatic adjustment."""
        if self._auto_mode and self._startup_complete:
            await self.update_automatic_temperature()
            self.async_write_ha_state()

    async def update_automatic_temperature(self):
        """MAIN METHOD: Update target temperature automatically based on weather and Sonoff sensor."""
        if not self._weather_manager or not self._auto_mode:
            return

        # Get current room temperature from Sonoff sensor (if available)
        current_room_temp = None
        if self._sonoff_sensor:
            try:
                sonoff_state = self.hass.states.get(self._sonoff_sensor)
                if sonoff_state and sonoff_state.state not in ['unknown', 'unavailable']:
                    current_room_temp = float(sonoff_state.state)
                    _LOGGER.debug(f"Using Sonoff sensor {self._sonoff_sensor}: {current_room_temp}°C")
            except (ValueError, TypeError):
                _LOGGER.warning(f"Invalid temperature reading from Sonoff sensor {self._sonoff_sensor}")

        # If no Sonoff sensor, fall back to Roth sensor
        if current_room_temp is None:
            current_room_temp = self.current_temperature
            _LOGGER.debug(f"Using Roth sensor: {current_room_temp}°C")

        # Calculate the new automatic target temperature
        new_target = self._weather_manager.calculate_target_temperature(
            self._base_comfort_temp,
            current_room_temp
        )

        # Get current target from Roth device
        current_target = self.target_temperature

        if abs(new_target - current_target) >= 0.2:  # Only update if significant change
            _LOGGER.info(
                f"Auto-adjusting {self.name}: {current_room_temp}°C → {new_target}°C "
                f"(was {current_target}°C)"
            )
            await self.hass.async_add_executor_job(
                self.coordinator.devices[self.device_id].set_target_temperature, new_target
            )
            await self.coordinator.async_request_refresh()

    async def async_set_preset_mode(self, preset_mode):
        """Set preset mode - Auto Weather enables automatic control."""
        self._auto_mode = preset_mode == "Auto Weather"

        if not self._auto_mode:
            # Set standard preset mode
            mode_settings = PRESET_MODES[preset_mode]
            device = self.coordinator.devices[self.device_id]
            await self.hass.async_add_executor_job(device.set_operation_mode, mode_settings.mode)
            await self.hass.async_add_executor_job(device.set_week_program, mode_settings.program)
            _LOGGER.info(f"{self.name}: Switched to manual preset '{preset_mode}'")
        else:
            # Enable automatic control
            if self._weather_manager:
                await self.update_automatic_temperature()
            _LOGGER.info(f"{self.name}: Enabled automatic weather-based control")

        await self.coordinator.async_request_refresh()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature - this updates the base comfort temperature."""
        if kwargs.get(ATTR_TEMPERATURE) is not None:
            target_temperature = kwargs.get(ATTR_TEMPERATURE)

            if self._auto_mode:
                # Update base comfort temperature for automatic calculations
                old_comfort = self._base_comfort_temp
                self._base_comfort_temp = target_temperature
                _LOGGER.info(f"{self.name}: Base comfort temp updated from {old_comfort}°C to {self._base_comfort_temp}°C")
                # Immediately recalculate with new base
                await self.update_automatic_temperature()
            else:
                # Manual mode - set temperature directly
                await self.hass.async_add_executor_job(
                    self.coordinator.devices[self.device_id].set_target_temperature,
                    target_temperature
                )
                _LOGGER.info(f"{self.name}: Manual temperature set to {target_temperature}°C")

            await self.coordinator.async_request_refresh()

    @property
    def current_temperature(self):
        """Return the current temperature from Sonoff sensor (preferred) or Roth sensor (fallback)."""
        # Try Sonoff sensor first
        if self._sonoff_sensor:
            try:
                sonoff_state = self.hass.states.get(self._sonoff_sensor)
                if sonoff_state and sonoff_state.state not in ['unknown', 'unavailable']:
                    return float(sonoff_state.state)
            except (ValueError, TypeError):
                pass

        # Fallback to Roth sensor
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
        if self._auto_mode:
            return "Auto Weather"

        device = self.coordinator.devices[self.device_id]
        operation_mode = device.get_operation_mode()
        week_program = device.get_week_program()
        return TOUCHLINE_HA_PRESETS.get((operation_mode, week_program))

    @property
    def preset_modes(self):
        """Return available preset modes."""
        modes = list(PRESET_MODES)
        if not self._weather_adaptation:
            modes.remove("Auto Weather")
        return modes

    @property
    def name(self):
        """Return the name of the climate device."""
        device_id_plus_one = self.device_id + 1
        if device_id_plus_one in self._names:
            return self._names[device_id_plus_one]
        return self._name_template.format(id=device_id_plus_one)

    @property
    def device_info(self):
        """Return device information about this entity."""
        return {
            "identifiers": {(DOMAIN, f"touchline_auto_{self.device_id}")},
            "name": self.name,
            "manufacturer": "Roth + Sonoff",
            "model": "Touchline Auto",
        }

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self.coordinator.last_update_success

    @property
    def extra_state_attributes(self):
        """Return additional state attributes."""
        attrs = {}
        
        # Add Sonoff sensor info
        if self._sonoff_sensor:
            attrs["sonoff_sensor"] = self._sonoff_sensor
            
        # Add weather forecast info
        if self._weather_manager and self._weather_manager.forecast_temp_3h is not None:
            attrs["forecast_3h"] = self._weather_manager.forecast_temp_3h
            attrs["base_comfort_temp"] = self._base_comfort_temp
            
        attrs["auto_mode"] = self._auto_mode
        return attrs

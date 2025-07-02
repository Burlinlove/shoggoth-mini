"""Orchestrator configuration for the main application."""

import os
import logging
from pydantic import Field
from .base import BaseConfig

logger = logging.getLogger(__name__)


class OrchestratorConfig(BaseConfig):
    """Configuration for the orchestrator application."""

    # OpenAI API Configuration
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key for Realtime API access (loads from OPENAI_API_KEY env var)",
    )
    websocket_url: str = Field(
        default="wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
        description="WebSocket URL for OpenAI Realtime API",
    )
    websocket_headers: list[str] = Field(
        default_factory=lambda: [
            "Authorization: Bearer YOUR_API_KEY_HERE",
            "OpenAI-Beta: realtime=v1",
        ],
        description="WebSocket headers for OpenAI Realtime API connection",
    )

    # Audio Configuration
    audio_sample_rate: int = Field(
        default=16_000, description="Audio sample rate in Hz"
    )
    audio_channels: int = Field(default=1, description="Number of audio channels")
    audio_dtype: str = Field(default="int16", description="Audio data type")
    audio_block_size: int = Field(
        default=2_048, description="Audio block size for processing"
    )

    # Visual Configuration
    wave_detection_depth_z_max: float = Field(
        default=-0.40,
        description="Depth threshold for wave detection (Z distance)",
    )

    # Finger-following configuration
    finger_follow_z_threshold: float = Field(
        default=-0.14,
        description="Depth threshold (Z) that defines when a finger is considered 'near' for finger-following",
    )
    finger_follow_y_threshold: float = Field(
        default=0.25,
        description="Maximum Y height threshold for finger-following (finger must be below this height)",
    )

    # Timing Configuration
    idle_start_delay_seconds: float = Field(
        default=2.0, description="Delay before starting idle motion after speech stops"
    )

    # System Messages and Tools
    system_prompt: str = Field(
        description="System prompt for the AI assistant",
    )

    def get_websocket_headers(self) -> list[str]:
        """Get WebSocket headers with proper API key integration."""
        return [
            f"Authorization: Bearer {self.openai_api_key}",
            "OpenAI-Beta: realtime=v1",
        ]

    def get_tools_definition(self) -> list:
        """Get tool definitions for OpenAI function calling."""
        return [
            {
                "type": "function",
                "name": "perform_primitive",
                "description": (
                    "Performs a motion primitive to express the assistant's current physical state in response to the user's input. "
                    "Available primitives: <yes>, <no>, <shake>, <circle>, <grab_object>, <release_object>, <high_five>"
                    "Use <yes> for agreement/understanding, eg. when user asks you a question. "
                    "Use <no> for disagreement/confusion, eg. when user asks you a question. "
                    "Use <shake> for waving your hand, to say hi or goodbye or similar. "
                    "Use <circle> for expressing excitement or happiness. "
                    "Use <grab_object> when the user asks you to grab or hold his finger or an object. "
                    "Use <release_object> when the user asks you to release. Do not release if the user is not asking you to release explicitly. "
                    "Use <high_five> when the user asks for a high five or when celebrating something together."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The primitive token to execute (<yes>, <no>, <shake>, <circle>, <grab_object>, <release_object>).",
                            "enum": [
                                "<yes>",
                                "<no>",
                                "<shake>",
                                "<circle>",
                                "<grab_object>",
                                "<release_object>",
                                "<high_five>",
                            ],
                        },
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "stay_silent",
                "description": "Use this function to give the user an opportunity to finish their thought.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "follow_finger",
                "description": (
                    "Starts a closed-loop policy that makes the robot continuously track the user's fingertip until stopped. "
                    "Call this when you receive a <finger near> visual cue. Do not call it if the user is asking you to grab or hold something, this is a separate tool."
                    "Do not call it if the policy is already running."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]

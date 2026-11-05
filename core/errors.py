"""Errors safe to present without leaking provider responses or credentials."""


class AssistantError(Exception):
    """A recoverable configuration, input, or provider error."""


class ConfigurationError(AssistantError):
    pass


class NoEligibleModel(AssistantError):
    pass


class ProviderError(AssistantError):
    pass


class MediaError(AssistantError):
    pass

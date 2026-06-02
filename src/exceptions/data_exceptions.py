class DataException(Exception):
    """Base exception for data-related errors."""
    pass


class ConfigLoadException(DataException):
    pass


class IngestionException(DataException):
    pass


class ValidationException(DataException):
    pass

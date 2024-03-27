import time

_logged = set()
_disabled = False
_periodic_log = False
_last_logged = 0.0


def log_once(key):
    """Returns True if this is the "first" call for a given key.

    Various logging settings can adjust the definition of "first".

    Example:
        >>> if log_once("some_key"):
        ...     logger.info("Some verbose logging statement")

    :param key: the key to check if it has been logged before.
    """

    global _last_logged

    if _disabled:
        return False
    elif key not in _logged:
        _logged.add(key)
        _last_logged = time.time()
        return True
    elif _periodic_log and time.time() - _last_logged > 60.0:
        _logged.clear()
        _last_logged = time.time()
        return False
    else:
        return False


def override(cls):
    """Decorator for documenting method overrides.

    :param cls: the superclass that provides the overridden method. If this cls does not
        actually have the method, an error is raised.
    """

    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError(f"{method} does not override any method of {cls}")
        return method

    return check_override

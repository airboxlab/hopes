import re
from datetime import datetime, timedelta, timezone
from re import Pattern
from zoneinfo import ZoneInfo


def should_skip_day(day_str: str, excluded_days: list[str], weekdays_only: bool = True) -> bool:
    """Returns True if the day must be skipped (weekend or blacklisted).

    day_str is 'YYYY-MM-DD'.
    """
    d = datetime.strptime(day_str, "%Y-%m-%d").date()
    if weekdays_only and d.weekday() >= 5:
        return True
    return day_str in excluded_days


# Reorganizes raw trajectory files on S3 by local day.
# Filenames contain UTC timestamps, so we parse them, convert to local timezone (Europe/Paris), and move each file
# into the corresponding local-day folder.


def daterange_inclusive(start_day: str, end_day: str):
    d0 = datetime.strptime(start_day, "%Y-%m-%d").date()
    d1 = datetime.strptime(end_day, "%Y-%m-%d").date()
    assert d0 <= d1, "START_DAY must be <= END_DAY"
    d = d0
    while d <= d1:
        yield d.isoformat()
        d += timedelta(days=1)


def local_day_from_filename_utc(key: str, regex_key: Pattern, local_tz: str) -> str | None:
    name = key.split("/")[-1]
    m = regex_key.match(name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    dt_utc = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    dt_local = dt_utc.astimezone(ZoneInfo(local_tz))
    return dt_local.date().isoformat()


def day_minus_one(day_str: str) -> str:
    d = datetime.strptime(day_str, "%Y-%m-%d").date()
    return (d - timedelta(days=1)).isoformat()


def extract_end_date_from_episode_id(episode_id: str) -> str:
    """Extract the trailing date from 'episode_id' values.

    Expected format: ..._YYYY_MM_DD  -> returns 'YYYY-MM-DD'
    Example: BATIMENT_..._2026_02_20 -> '2026-02-20'
    """
    if not isinstance(episode_id, str):
        raise ValueError(f"episode_id is not a string: {episode_id!r}")

    # We explicitly anchor to end-of-string to avoid matching earlier underscores
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})$", episode_id)
    if not m:
        raise ValueError(f"Could not parse date from episode_id: {episode_id!r}")

    yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
    return f"{yyyy}-{mm}-{dd}"


def make_prefix_new(run_id: str, model_name: str, base_training_prefix: str) -> str:
    return f"{base_training_prefix}/{run_id}/config/" f"{model_name}/output/" f"{model_name}"

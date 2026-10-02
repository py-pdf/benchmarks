"""Look up when a package version was published on PyPI."""

import requests


def get_release_date(package: str, version: str) -> str:
    """Return the ISO date (YYYY-MM-DD) a package version was published on PyPI.

    Returns an empty string if the lookup fails, e.g. due to no network
    connectivity, an unknown package, or an unknown version.
    """
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        urls = response.json()["urls"]
        if not urls:
            return ""
        upload_time = min(entry["upload_time_iso_8601"] for entry in urls)
        return upload_time[:10]
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return ""

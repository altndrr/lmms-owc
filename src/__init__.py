def _package_metadata_as_dict(name: str | None = None, exclude: list | None = None) -> dict:
    """Get (distribution) package metadata and return it as a dict.

    Args:
    ----
        name (str, optional): The name of the distribution package.
        exclude (list, optional): A list of keys to exclude from the metadata.

    """
    from collections import Counter
    from importlib.metadata import PackageNotFoundError, metadata

    try:
        m = metadata(name or __name__)
    except PackageNotFoundError:
        return {}

    out = {}
    for key, count in Counter(m).items():
        if key in exclude:
            continue

        if count == 1:
            out[key] = m[key]
        elif count > 1:
            out[key] = [val for val in m.get_all(key)]

    # post-process project URLs
    if "Project-URL" in out:
        if isinstance(out["Project-URL"], str):
            out["Project-URL"] = [out["Project-URL"]]

        out["Project-URL"] = {
            url.split(", ")[0].strip(): url.split(", ")[1].strip() for url in out["Project-URL"]
        }

    return out


__meta__ = _package_metadata_as_dict(name="lmms-owc", exclude=["License", "Classifier"])
__author__ = __meta__.get("Author-email", None)
__license__ = "MIT"
__maintainer__ = __author__
__version__ = __meta__.get("Version", None)
__website__ = __meta__.get("Project-URL", {}).get("Repository", None)

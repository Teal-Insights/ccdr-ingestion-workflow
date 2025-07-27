# Use remove element by id to remove unused clip paths for better reliability than regex

import tempfile
import subprocess
import os
from PIL import Image
from typing import Literal
import re
import io
import svgelements
import numpy as np


def has_geometry(element: svgelements.SVGElement) -> bool:
    """
    Check if an SVG element is visible and has geometry (e.g., not just a line).

    An element is considered to have geometry if its bounding box has a positive
    width and height.

    Args:
        element: The SVG element to check.

    Returns:
        True if the element is visible and has geometry, False otherwise.
    """
    # 1. Check for visibility attributes that would make it invisible
    try:
        if hasattr(element, "values"):
            if element.values.get("visibility") == "hidden":
                return False
            if element.values.get("display") == "none":
                return False
    except (KeyError, AttributeError):
        pass  # Some elements might not have 'values'

    # 2. Check for geometry by inspecting the bounding box
    if hasattr(element, "bbox") and callable(element.bbox):
        try:
            element_bbox = element.bbox()
            if element_bbox is not None:
                # bbox is typically (x0, y0, x1, y1)
                x0, y0, x1, y1 = element_bbox

                # Ensure all values are numeric
                x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)

                # Check for positive width and height to exclude lines or points
                if x1 > x0 and y1 > y0:
                    return True
        except (ValueError, TypeError, IndexError):
            # Handles cases where bbox is malformed or not convertible to floats.
            return False

    return False


def is_font_element(element, font_prefix: str = "font_") -> bool:
    """Return True if the element is a font glyph definition or a <use> clone
    that references a glyph whose id starts with ``font_prefix``.

    It works by checking:
    1. The element's own ``id``.
    2. Any *href* / *xlink:href* attributes that reference another element id.
    """
    # 1. Direct id check
    element_id = getattr(element, "id", None)
    if element_id and str(element_id).startswith(font_prefix):
        return True

    # 2. Inspect values / nested attribute dictionaries for hrefs referencing glyphs
    if hasattr(element, "values") and isinstance(element.values, dict):
        href_keys = (
            "{http://www.w3.org/1999/xlink}href",
            "xlink:href",
            "href",
        )

        # a) direct keys in values
        for key in href_keys:
            ref = element.values.get(key)
            if ref and isinstance(ref, str) and ref.startswith("#"):
                if ref.lstrip("#").startswith(font_prefix):
                    return True

        # b) nested "attributes" dict present in some parsed elements
        attrs = element.values.get("attributes")
        if isinstance(attrs, dict):
            for key in href_keys:
                ref = attrs.get(key)
                if ref and isinstance(ref, str) and ref.startswith("#"):
                    if ref.lstrip("#").startswith(font_prefix):
                        return True

    return False

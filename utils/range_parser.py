def parse_range_string(range_str: str) -> list[int]:
    """
    Parse a comma-separated range string into a list of integers.
    
    Args:
        range_str: String like "0-3,5,7-9" or empty string
        
    Returns:
        List of integers representing all tag ids in the ranges
        
    Raises:
        ValueError: If the range string format is invalid
    """
    if not range_str.strip():
        return []
    
    ids: list[int] = []
    parts = range_str.split(",")
    
    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range like "1-3"
            range_parts = part.split("-", 1)
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range format: {part}")
            try:
                start_num = int(range_parts[0].strip())
                end_num = int(range_parts[1].strip())
                if start_num > end_num:
                    raise ValueError(f"Invalid range: start ({start_num}) > end ({end_num})")
                ids.extend(range(start_num, end_num + 1))
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Non-numeric values in range: {part}")
                raise
        else:
            # Handle single number like "5"
            try:
                ids.append(int(part))
            except ValueError:
                raise ValueError(f"Non-numeric value: {part}")
    
    return sorted(ids)
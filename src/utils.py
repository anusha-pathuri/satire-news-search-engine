def load_txt(filepath: str) -> list[str]:
    """Load a list of strings from a .txt file line-by-line."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f: 
        lines = f.readlines()
        for line in lines:
            data.append(line.strip())
    
    return data


def save_txt(data: list, filepath: str) -> None:
    """Save a list of values to a .txt file line-by-line."""
    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            f.write(str(line) + "\n")

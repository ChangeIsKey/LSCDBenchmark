
def overrides(d: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in d.items()]

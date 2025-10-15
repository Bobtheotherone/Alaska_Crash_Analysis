#config for global variables across project

UNKNOWN_THRESHOLD = 10.0  #percentage threshold for unknowns in a column
YES_NO_THRESHOLD = 1.0  #percentage threshold for yes/no in a column


#verify they are valid percentages
def validate():
    def _ok(x): return isinstance(x, (int, float)) and 0 <= x <= 100
    if not _ok(UNKNOWN_THRESHOLD):
        raise ValueError(f"UNKNOWN_THRESHOLD must be 0–100, got {UNKNOWN_THRESHOLD}")
    if not _ok(YES_NO_THRESHOLD):
        raise ValueError(f"YES_NO_THRESHOLD must be 0–100, got {YES_NO_THRESHOLD}")
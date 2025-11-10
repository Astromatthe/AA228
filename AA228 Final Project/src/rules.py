from typing import List

def count_face_total(
        dice: List[List[int]],
        face: int
) -> int:
    """Counts the total number of faces showing in the given dice."""
    return int((dice == face).sum())
import numpy as np
import numpy.typing as npt

def dot_product(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    return a.dot(b)
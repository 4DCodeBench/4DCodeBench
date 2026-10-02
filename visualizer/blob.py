"""The binary buffer the browser reads arrays from, and the manifest that names them."""

import numpy as np


class Blob:
    """A growing binary file plus the manifest that names the arrays in it."""

    def __init__(self):
        self.parts, self.manifest, self.offset = [], {}, 0

    def add(self, name, array, dtype=np.float32):
        array = np.ascontiguousarray(array, dtype=dtype)
        self.manifest[name] = {"offset": self.offset, "shape": list(array.shape),
                               "dtype": np.dtype(dtype).name}
        self.parts.append(array.tobytes())
        self.offset += array.nbytes
        # a typed array must start on a multiple of its item size
        if self.offset % 4:
            padding = 4 - self.offset % 4
            self.parts.append(b"\0" * padding)
            self.offset += padding
        return self.manifest[name]

    def bytes(self):
        return b"".join(self.parts)

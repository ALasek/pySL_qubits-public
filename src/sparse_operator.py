import numbers

import cupy as cp


class SumSparseOperator:
    def __init__(self, terms, shape, dtype):
        self.terms = list(terms)
        self.shape = shape
        self.dtype = dtype

    @property
    def nnz(self):
        return sum(getattr(term, "nnz", 0) for term in self.terms)

    def __matmul__(self, other):
        result = None
        for term in self.terms:
            value = term @ other
            if result is None:
                result = value
            else:
                result += value
        if result is not None:
            return result

        out_shape = (self.shape[0],) if other.ndim == 1 else (self.shape[0], other.shape[1])
        return cp.zeros(out_shape, dtype=self.dtype)

    def __add__(self, other):
        if _is_zero_scalar(other):
            return self
        if isinstance(other, SumSparseOperator):
            return SumSparseOperator(self.terms + other.terms, self.shape, self.dtype)
        if hasattr(other, "shape"):
            return SumSparseOperator(self.terms + [other], self.shape, self.dtype)
        return NotImplemented

    def __radd__(self, other):
        if _is_zero_scalar(other):
            return self
        if hasattr(other, "shape"):
            return SumSparseOperator([other] + self.terms, self.shape, self.dtype)
        return NotImplemented

    def toarray(self):
        result = cp.zeros(self.shape, dtype=self.dtype)
        for term in self.terms:
            result += term.toarray()
        return result


def _is_zero_scalar(value):
    return isinstance(value, numbers.Number) and value == 0


class HybridSparseOperator:
    def __init__(self, diagonal, terms, shape, dtype):
        self.diagonal = diagonal
        self.terms = list(terms)
        self.shape = shape
        self.dtype = dtype

    @property
    def nnz(self):
        diag_nnz = int(cp.count_nonzero(self.diagonal).item()) if self.diagonal is not None else 0
        return diag_nnz + sum(getattr(term, "nnz", 0) for term in self.terms)

    def __matmul__(self, other):
        result = None
        if self.diagonal is not None:
            if other.ndim == 1:
                result = self.diagonal * other
            else:
                result = self.diagonal[:, None] * other

        for term in self.terms:
            value = term @ other
            if result is None:
                result = value
            else:
                result += value

        if result is not None:
            return result

        out_shape = (self.shape[0],) if other.ndim == 1 else (self.shape[0], other.shape[1])
        return cp.zeros(out_shape, dtype=self.dtype)

    def __add__(self, other):
        if _is_zero_scalar(other):
            return self
        if isinstance(other, HybridSparseOperator):
            diagonal = _add_diagonals(self.diagonal, other.diagonal, self.shape[0], self.dtype)
            return HybridSparseOperator(diagonal, self.terms + other.terms, self.shape, self.dtype)
        if isinstance(other, SumSparseOperator):
            return HybridSparseOperator(self.diagonal, self.terms + other.terms, self.shape, self.dtype)
        if hasattr(other, "shape"):
            return HybridSparseOperator(self.diagonal, self.terms + [other], self.shape, self.dtype)
        return NotImplemented

    def __radd__(self, other):
        if _is_zero_scalar(other):
            return self
        if hasattr(other, "shape"):
            return HybridSparseOperator(self.diagonal, [other] + self.terms, self.shape, self.dtype)
        return NotImplemented

    def toarray(self):
        result = cp.zeros(self.shape, dtype=self.dtype)
        if self.diagonal is not None:
            result[cp.arange(self.shape[0]), cp.arange(self.shape[0])] = self.diagonal
        for term in self.terms:
            result += term.toarray()
        return result


def _add_diagonals(left, right, size, dtype):
    if left is None:
        return right
    if right is None:
        return left
    return left + right

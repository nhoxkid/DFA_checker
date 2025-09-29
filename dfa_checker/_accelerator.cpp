#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <stdexcept>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace {

inline int index_for_bit(unsigned long long bit) {
#ifdef _MSC_VER
    unsigned long idx = 0;
    _BitScanForward64(&idx, bit);
    return static_cast<int>(idx);
#else
    return static_cast<int>(__builtin_ctzll(bit));
#endif
}

inline unsigned long long as_ull(PyObject* number) {
    const unsigned long long value = PyLong_AsUnsignedLongLongMask(number);
    if (PyErr_Occurred()) {
        throw std::runtime_error("expected non-negative integer mask");
    }
    return value;
}

inline void ensure_width(Py_ssize_t count) {
    if (count > 63) {
        throw std::runtime_error("accelerator only handles <= 63 states");
    }
}

}  // namespace

static PyObject* accel_compute_epsilon_closures(PyObject*, PyObject* args) {
    PyObject* matrix_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &matrix_obj)) {
        return nullptr;
    }
    PyObject* seq = PySequence_Fast(matrix_obj, "epsilon matrix must be a sequence");
    if (!seq) {
        return nullptr;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    try {
        ensure_width(count);
        std::vector<unsigned long long> matrix(static_cast<size_t>(count));
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
            matrix[static_cast<size_t>(i)] = as_ull(item);
        }
        PyObject* result = PyTuple_New(count);
        if (!result) {
            Py_DECREF(seq);
            return nullptr;
        }
        for (Py_ssize_t i = 0; i < count; ++i) {
            unsigned long long visited = 1ULL << i;
            std::vector<int> stack;
            stack.push_back(static_cast<int>(i));
            while (!stack.empty()) {
                const int state = stack.back();
                stack.pop_back();
                unsigned long long mask = matrix[static_cast<size_t>(state)];
                while (mask) {
                    const unsigned long long bit = mask & -mask;
                    mask ^= bit;
                    const int idx = index_for_bit(bit);
                    const unsigned long long flag = 1ULL << idx;
                    if (!(visited & flag)) {
                        visited |= flag;
                        stack.push_back(idx);
                    }
                }
            }
            PyObject* value = PyLong_FromUnsignedLongLong(visited);
            if (!value) {
                Py_DECREF(result);
                Py_DECREF(seq);
                return nullptr;
            }
            PyTuple_SET_ITEM(result, i, value);
        }
        Py_DECREF(seq);
        return result;
    } catch (const std::runtime_error& err) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, err.what());
        return nullptr;
    }
}

static PyObject* accel_build_symbol_matrix(PyObject*, PyObject* args) {
    PyObject* transition_obj = nullptr;
    PyObject* closures_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &transition_obj, &closures_obj)) {
        return nullptr;
    }
    PyObject* rows = PySequence_Fast(transition_obj, "transition matrix must be a sequence");
    if (!rows) {
        return nullptr;
    }
    PyObject* closures = PySequence_Fast(closures_obj, "closures must be a sequence");
    if (!closures) {
        Py_DECREF(rows);
        return nullptr;
    }
    const Py_ssize_t state_count = PySequence_Fast_GET_SIZE(rows);
    try {
        ensure_width(state_count);
        std::vector<unsigned long long> closure_masks(static_cast<size_t>(state_count));
        for (Py_ssize_t i = 0; i < state_count; ++i) {
            closure_masks[static_cast<size_t>(i)] = as_ull(PySequence_Fast_GET_ITEM(closures, i));
        }
        PyObject* outer = PyTuple_New(state_count);
        if (!outer) {
            Py_DECREF(rows);
            Py_DECREF(closures);
            return nullptr;
        }
        Py_ssize_t symbol_count = 0;
        for (Py_ssize_t state = 0; state < state_count; ++state) {
            PyObject* row_src = PySequence_Fast(PySequence_Fast_GET_ITEM(rows, state), "row must be sequence");
            if (!row_src) {
                Py_DECREF(outer);
                Py_DECREF(rows);
                Py_DECREF(closures);
                return nullptr;
            }
            if (state == 0) {
                symbol_count = PySequence_Fast_GET_SIZE(row_src);
            }
            PyObject* new_row = PyTuple_New(symbol_count);
            if (!new_row) {
                Py_DECREF(row_src);
                Py_DECREF(outer);
                Py_DECREF(rows);
                Py_DECREF(closures);
                return nullptr;
            }
            for (Py_ssize_t symbol = 0; symbol < symbol_count; ++symbol) {
                unsigned long long mask = as_ull(PySequence_Fast_GET_ITEM(row_src, symbol));
                unsigned long long closed = 0;
                while (mask) {
                    const unsigned long long bit = mask & -mask;
                    mask ^= bit;
                    const int idx = index_for_bit(bit);
                    closed |= closure_masks[static_cast<size_t>(idx)];
                }
                PyObject* value = PyLong_FromUnsignedLongLong(closed);
                if (!value) {
                    Py_DECREF(row_src);
                    Py_DECREF(new_row);
                    Py_DECREF(outer);
                    Py_DECREF(rows);
                    Py_DECREF(closures);
                    return nullptr;
                }
                PyTuple_SET_ITEM(new_row, symbol, value);
            }
            Py_DECREF(row_src);
            PyTuple_SET_ITEM(outer, state, new_row);
        }
        Py_DECREF(rows);
        Py_DECREF(closures);
        return outer;
    } catch (const std::runtime_error& err) {
        Py_DECREF(rows);
        Py_DECREF(closures);
        PyErr_SetString(PyExc_ValueError, err.what());
        return nullptr;
    }
}

static PyObject* accel_subset_step(PyObject*, PyObject* args) {
    PyObject* symbol_matrix = nullptr;
    unsigned long long subset_mask = 0;
    Py_ssize_t symbol_index = 0;
    if (!PyArg_ParseTuple(args, "OKn", &symbol_matrix, &subset_mask, &symbol_index)) {
        return nullptr;
    }
    PyObject* rows = PySequence_Fast(symbol_matrix, "symbol matrix must be a sequence");
    if (!rows) {
        return nullptr;
    }
    const Py_ssize_t state_count = PySequence_Fast_GET_SIZE(rows);
    unsigned long long result = 0;
    try {
        ensure_width(state_count);
        while (subset_mask) {
            const unsigned long long bit = subset_mask & -subset_mask;
            subset_mask ^= bit;
            const int idx = index_for_bit(bit);
            if (idx >= state_count) {
                Py_DECREF(rows);
                PyErr_SetString(PyExc_ValueError, "subset state index out of range");
                return nullptr;
            }
            PyObject* row_src = PySequence_Fast(PySequence_Fast_GET_ITEM(rows, idx), "row must be sequence");
            if (!row_src) {
                Py_DECREF(rows);
                return nullptr;
            }
            const Py_ssize_t row_len = PySequence_Fast_GET_SIZE(row_src);
            if (symbol_index < 0 || symbol_index >= row_len) {
                Py_DECREF(row_src);
                Py_DECREF(rows);
                PyErr_SetString(PyExc_ValueError, "symbol index out of range");
                return nullptr;
            }
            result |= as_ull(PySequence_Fast_GET_ITEM(row_src, symbol_index));
            Py_DECREF(row_src);
        }
        Py_DECREF(rows);
        return PyLong_FromUnsignedLongLong(result);
    } catch (const std::runtime_error& err) {
        Py_DECREF(rows);
        PyErr_SetString(PyExc_ValueError, err.what());
        return nullptr;
    }
}

static PyMethodDef AcceleratorMethods[] = {
    {"compute_epsilon_closures", reinterpret_cast<PyCFunction>(accel_compute_epsilon_closures), METH_VARARGS, "fast epsilon closures"},
    {"build_symbol_matrix", reinterpret_cast<PyCFunction>(accel_build_symbol_matrix), METH_VARARGS, "crunch symbol closure matrix"},
    {"subset_step", reinterpret_cast<PyCFunction>(accel_subset_step), METH_VARARGS, "advance subset mask"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef AcceleratorModule = {
    PyModuleDef_HEAD_INIT,
    "_accelerator",
    "bitmask kungfu",
    -1,
    AcceleratorMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC PyInit__accelerator(void) {
    return PyModule_Create(&AcceleratorModule);
}

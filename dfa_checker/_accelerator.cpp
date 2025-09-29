#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace {

constexpr Py_ssize_t kMaxStates = 63;

inline int bit_index(std::uint64_t bit) {
#if defined(__GNUC__) && defined(__x86_64__)
    unsigned long idx;
    asm("bsf %1, %0" : "=r"(idx) : "r"(bit));
    return static_cast<int>(idx);
#elif defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward64(&idx, bit);
    return static_cast<int>(idx);
#else
    return static_cast<int>(__builtin_ctzll(bit));
#endif
}

inline std::uint64_t mask_from_python(PyObject* obj) {
    const std::uint64_t value = PyLong_AsUnsignedLongLongMask(obj);
    if (PyErr_Occurred()) {
        throw std::runtime_error("expected non-negative integer mask");
    }
    return value;
}

inline unsigned int best_thread_count(std::size_t work_items) {
    unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    if (work_items == 0) {
        return 1;
    }
    if (work_items < hw) {
        return static_cast<unsigned int>(work_items);
    }
    return hw;
}

inline std::uint64_t compute_closure(
    Py_ssize_t state_idx,
    const std::vector<std::uint64_t>& matrix
) {
    std::uint64_t visited = 1ULL << state_idx;
    std::vector<int> stack;
    stack.reserve(16);
    stack.push_back(static_cast<int>(state_idx));

    while (!stack.empty()) {
        const int here = stack.back();
        stack.pop_back();
        std::uint64_t mask = matrix[static_cast<std::size_t>(here)];
        while (mask) {
            const std::uint64_t bit = mask & -mask;
            mask ^= bit;
            const int next = bit_index(bit);
            const std::uint64_t flag = 1ULL << next;
            if ((visited & flag) == 0) {
                visited |= flag;
                stack.push_back(next);
            }
        }
    }
    return visited;
}

inline void fast_or(std::uint64_t& accumulator, std::uint64_t mask) {
    accumulator |= mask;
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
    if (count > kMaxStates) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "accelerator supports up to 63 states");
        return nullptr;
    }

    std::vector<std::uint64_t> matrix(static_cast<std::size_t>(count));
    try {
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
            matrix[static_cast<std::size_t>(i)] = mask_from_python(item);
        }
    } catch (const std::runtime_error& err) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, err.what());
        return nullptr;
    }

    std::vector<std::uint64_t> closures(static_cast<std::size_t>(count));
    const unsigned int thread_count = best_thread_count(static_cast<std::size_t>(count));
    const Py_ssize_t chunk = thread_count ? (count + thread_count - 1) / thread_count : count;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    Py_BEGIN_ALLOW_THREADS
    for (unsigned int t = 0; t < thread_count; ++t) {
        const Py_ssize_t start = static_cast<Py_ssize_t>(t) * chunk;
        if (start >= count) {
            break;
        }
        const Py_ssize_t end = std::min<Py_ssize_t>(count, start + chunk);
        workers.emplace_back([
            start,
            end,
            &matrix,
            &closures
        ]() {
            for (Py_ssize_t idx = start; idx < end; ++idx) {
                closures[static_cast<std::size_t>(idx)] = compute_closure(idx, matrix);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    Py_END_ALLOW_THREADS

    PyObject* result = PyTuple_New(count);
    if (!result) {
        Py_DECREF(seq);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* value = PyLong_FromUnsignedLongLong(closures[static_cast<std::size_t>(i)]);
        if (!value) {
            Py_DECREF(seq);
            Py_DECREF(result);
            return nullptr;
        }
        PyTuple_SET_ITEM(result, i, value);
    }
    Py_DECREF(seq);
    return result;
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
    PyObject* closures_seq = PySequence_Fast(closures_obj, "closures must be a sequence");
    if (!closures_seq) {
        Py_DECREF(rows);
        return nullptr;
    }

    const Py_ssize_t state_count = PySequence_Fast_GET_SIZE(rows);
    if (state_count > kMaxStates) {
        Py_DECREF(rows);
        Py_DECREF(closures_seq);
        PyErr_SetString(PyExc_ValueError, "accelerator supports up to 63 states");
        return nullptr;
    }

    std::vector<std::uint64_t> closure_masks(static_cast<std::size_t>(state_count));
    try {
        for (Py_ssize_t i = 0; i < state_count; ++i) {
            closure_masks[static_cast<std::size_t>(i)] = mask_from_python(
                PySequence_Fast_GET_ITEM(closures_seq, i)
            );
        }
    } catch (const std::runtime_error& err) {
        Py_DECREF(rows);
        Py_DECREF(closures_seq);
        PyErr_SetString(PyExc_ValueError, err.what());
        return nullptr;
    }

    std::vector<std::vector<std::uint64_t>> transitions(static_cast<std::size_t>(state_count));
    Py_ssize_t symbol_count = 0;
    try {
        for (Py_ssize_t state = 0; state < state_count; ++state) {
            PyObject* row_obj = PySequence_Fast(
                PySequence_Fast_GET_ITEM(rows, state),
                "row must be a sequence"
            );
            if (!row_obj) {
                throw std::runtime_error("row conversion failed");
            }
            const Py_ssize_t row_len = PySequence_Fast_GET_SIZE(row_obj);
            if (state == 0) {
                symbol_count = row_len;
            } else if (row_len != symbol_count) {
                Py_DECREF(row_obj);
                throw std::runtime_error("row lengths must match");
            }
            std::vector<std::uint64_t> row(static_cast<std::size_t>(row_len));
            for (Py_ssize_t symbol = 0; symbol < row_len; ++symbol) {
                row[static_cast<std::size_t>(symbol)] = mask_from_python(
                    PySequence_Fast_GET_ITEM(row_obj, symbol)
                );
            }
            Py_DECREF(row_obj);
            transitions[static_cast<std::size_t>(state)] = std::move(row);
        }
    } catch (const std::runtime_error& err) {
        Py_DECREF(rows);
        Py_DECREF(closures_seq);
        PyErr_SetString(PyExc_ValueError, err.what());
        return nullptr;
    }

    std::vector<std::vector<std::uint64_t>> result(
        static_cast<std::size_t>(state_count),
        std::vector<std::uint64_t>(static_cast<std::size_t>(symbol_count), 0ULL)
    );

    const unsigned int thread_count = best_thread_count(static_cast<std::size_t>(state_count));
    const Py_ssize_t chunk = thread_count ? (state_count + thread_count - 1) / thread_count : state_count;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    Py_BEGIN_ALLOW_THREADS
    for (unsigned int t = 0; t < thread_count; ++t) {
        const Py_ssize_t start = static_cast<Py_ssize_t>(t) * chunk;
        if (start >= state_count) {
            break;
        }
        const Py_ssize_t end = std::min<Py_ssize_t>(state_count, start + chunk);
        workers.emplace_back([
            start,
            end,
            symbol_count,
            &transitions,
            &closure_masks,
            &result
        ]() {
            for (Py_ssize_t state = start; state < end; ++state) {
                auto& row = result[static_cast<std::size_t>(state)];
                const auto& src_row = transitions[static_cast<std::size_t>(state)];
                for (Py_ssize_t symbol = 0; symbol < symbol_count; ++symbol) {
                    std::uint64_t mask = src_row[static_cast<std::size_t>(symbol)];
                    std::uint64_t acc = 0ULL;
                    while (mask) {
                        const std::uint64_t bit = mask & -mask;
                        mask ^= bit;
                        const int idx = bit_index(bit);
                        fast_or(acc, closure_masks[static_cast<std::size_t>(idx)]);
                    }
                    row[static_cast<std::size_t>(symbol)] = acc;
                }
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    Py_END_ALLOW_THREADS

    PyObject* outer = PyTuple_New(state_count);
    if (!outer) {
        Py_DECREF(rows);
        Py_DECREF(closures_seq);
        return nullptr;
    }
    for (Py_ssize_t state = 0; state < state_count; ++state) {
        PyObject* row = PyTuple_New(symbol_count);
        if (!row) {
            Py_DECREF(rows);
            Py_DECREF(closures_seq);
            Py_DECREF(outer);
            return nullptr;
        }
        for (Py_ssize_t symbol = 0; symbol < symbol_count; ++symbol) {
            PyObject* value = PyLong_FromUnsignedLongLong(
                result[static_cast<std::size_t>(state)][static_cast<std::size_t>(symbol)]
            );
            if (!value) {
                Py_DECREF(rows);
                Py_DECREF(closures_seq);
                Py_DECREF(outer);
                Py_DECREF(row);
                return nullptr;
            }
            PyTuple_SET_ITEM(row, symbol, value);
        }
        PyTuple_SET_ITEM(outer, state, row);
    }

    Py_DECREF(rows);
    Py_DECREF(closures_seq);
    return outer;
}

static PyObject* accel_subset_step(PyObject*, PyObject* args) {
    PyObject* symbol_matrix = nullptr;
    unsigned long long subset_mask = 0ULL;
    Py_ssize_t symbol_index = 0;
    if (!PyArg_ParseTuple(args, "OKn", &symbol_matrix, &subset_mask, &symbol_index)) {
        return nullptr;
    }
    PyObject* rows = PySequence_Fast(symbol_matrix, "symbol matrix must be a sequence");
    if (!rows) {
        return nullptr;
    }
    const Py_ssize_t state_count = PySequence_Fast_GET_SIZE(rows);
    if (state_count > kMaxStates) {
        Py_DECREF(rows);
        PyErr_SetString(PyExc_ValueError, "accelerator supports up to 63 states");
        return nullptr;
    }

    std::uint64_t result = 0ULL;
    std::uint64_t temp = static_cast<std::uint64_t>(subset_mask);
    while (temp) {
        const std::uint64_t bit = temp & -temp;
        temp ^= bit;
        const int idx = bit_index(bit);
        if (idx >= state_count) {
            Py_DECREF(rows);
            PyErr_SetString(PyExc_ValueError, "subset state index out of range");
            return nullptr;
        }
        PyObject* row_obj = PySequence_Fast(
            PySequence_Fast_GET_ITEM(rows, idx),
            "row must be sequence"
        );
        if (!row_obj) {
            Py_DECREF(rows);
            return nullptr;
        }
        const Py_ssize_t row_len = PySequence_Fast_GET_SIZE(row_obj);
        if (symbol_index < 0 || symbol_index >= row_len) {
            Py_DECREF(row_obj);
            Py_DECREF(rows);
            PyErr_SetString(PyExc_ValueError, "symbol index out of range");
            return nullptr;
        }
        result |= mask_from_python(PySequence_Fast_GET_ITEM(row_obj, symbol_index));
        Py_DECREF(row_obj);
    }
    Py_DECREF(rows);
    return PyLong_FromUnsignedLongLong(result);
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

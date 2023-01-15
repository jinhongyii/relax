# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import inspect

import tvm
from tvm.contrib.cutlass.build import select_gemm_kernel, _get_cutlass_path
from tvm.contrib.cutlass.gen_gemm import CutlassGemmProfiler

GRAPH_PATTERN_CODE_LIST = dict()


def get_graph_pattern_cutlass_code(cutlass_op, *args, **kwargs):
    cutlass_op = [str(st) for st in cutlass_op]
    pattern = "/".join(cutlass_op)

    if pattern.startswith("dense"):
        arg_names = inspect.getargspec(cutlass_codegen_gemm)[0]
        return cutlass_codegen_gemm(*_attr_to_list(kwargs["attr"], arg_names))
    elif pattern.startswith("batch_dense"):
        arg_names = inspect.getargspec(cutlass_codegen_batch_gemm)[0]
        return cutlass_codegen_batch_gemm(*_attr_to_list(kwargs["attr"], arg_names))


def _convert_type_str(type):
    if isinstance(type, list):
        arr = []
        for t in type:
            arr.append(_convert_type_str(t))
        return arr
    elif isinstance(type, str):
        if type == "float16":
            return "cutlass::half_t"
        elif type == "float32":
            return "float"
    raise ValueError("type not supported")


def _convert_layout_str(layout):
    if isinstance(layout, list):
        arr = []
        for l in layout:
            arr.append(_convert_layout_str(l))
        return arr
    elif isinstance(layout, str):
        if layout == "row":
            return "cutlass::layout::RowMajor"
        elif layout == "col":
            return "cutlass::layout::ColumnMajor"
    raise ValueError("layout not supported")


def _reverse_layout(layout):
    if isinstance(layout, list):
        arr = []
        for l in layout:
            arr.append(_reverse_layout(l))
        return arr
    elif isinstance(layout, str):
        if layout == "cutlass::layout::RowMajor":
            return "cutlass::layout::ColumnMajor"
        elif layout == "cutlass::layout::ColumnMajor":
            return "cutlass::layout::RowMajor"
    raise ValueError("layout not supported")


def _attr_to_list(attr, arg_names):
    arr = []
    for n in arg_names:
        if n in attr:
            arr.append(attr[n])
    return arr


def cutlass_codegen_gemm(
    m, n, k, typea, typeb, typec, layouta, layoutb, layoutc, op_type, sm=80, bin_dir="./bin"
):
    cutlass_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), bin_dir)
    op_name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        m,
        k,
        n,
        typec,
        typea,
        typeb,
        layoutc,
        layouta,
        layoutb,
        False,
        False,
        False,
        True,
    )
    op_name = "Operation_" + op_name
    typea, typeb, typec = _convert_type_str([typea, typeb, typec])
    layouta, layoutb, layoutc = _convert_layout_str([layouta, layoutb, layoutc])
    r_layouta, r_layoutb, r_layoutc = _reverse_layout([layouta, layoutb, layoutc])

    if op_type == "cutlass.dense_bias" or op_type == "cutlass.dense_bias_relu":
        bias_param = "NDArray Bias, "
        bias_var_def = f"""
        {r_layoutc}::Stride::Index ld_bias(0);
        cutlass::half_t* bias = reinterpret_cast<cutlass::half_t*>(Bias->data);
        """
        bias_arg = "{bias, ld_bias}"
    else:
        bias_param = bias_var_def = ""
        bias_arg = "{c, ldc}"

    text = f"""
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {{

      using namespace tvm;
      using namespace tvm::runtime;

      void _GEMM(NDArray A, NDArray B, {bias_param}NDArray C) {{
        // A: [M, K], B: [K, N]
        CHECK_EQ(A->ndim, 2);
        CHECK_EQ(B->ndim, 2);
        CHECK_EQ(C->ndim, 2);
        CHECK_EQ(A->shape[1], B->shape[0]);
        int M = A->shape[0];
        int K = A->shape[1];
        int N = B->shape[1];
        CHECK_EQ(C->shape[0], M);
        CHECK_EQ(C->shape[1], N);
        // Define the GEMM operation
        {cutlass_op_def}
        {op_name} gemm_op;
        {typec} alpha(1.0);
        {typec} beta(0.0);
        {r_layouta}::Stride::Index lda(K);
        {r_layoutb}::Stride::Index ldb(N);
        {r_layoutc}::Stride::Index ldc(N);
        {bias_var_def}
        {typea}* a = reinterpret_cast<cutlass::half_t*>(A->data);
        {typeb}* b = reinterpret_cast<cutlass::half_t*>(B->data);
        {typec}* c = reinterpret_cast<cutlass::half_t*>(C->data);
        cutlass::Status status = gemm_op({{
            {{M, N, K}},     //
            {{a, lda}},      //
            {{b, ldb}},      //
            {bias_arg},      //
            {{c, ldc}},      //
            {{alpha, beta}}  //
        }});
        CHECK(status == cutlass::Status::kSuccess);
      }}

      }}  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({{global_symbol}}, _GEMM);
      """
    print(text)
    return text


def cutlass_codegen_batch_gemm(
    m, n, k, typea, typeb, typec, layouta, layoutb, layoutc, op_type, sm=80, bin_dir="./bin"
):
    cutlass_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), bin_dir)
    op_name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        m,
        k,
        n,
        typec,
        typea,
        typeb,
        layoutc,
        layouta,
        layoutb,
        False,
        True,
        False,
        True,
    )
    op_name = "Operation_" + op_name
    typea, typeb, typec = _convert_type_str([typea, typeb, typec])
    layouta, layoutb, layoutc = _convert_layout_str([layouta, layoutb, layoutc])
    r_layouta, r_layoutb, r_layoutc = _reverse_layout([layouta, layoutb, layoutc])

    if op_type == "cutlass.batch_matmul_bias" or op_type == "cutlass.batch_matmul_bias_relu":
        bias_param = "NDArray Bias, "
        bias_var_def = f"""
        {r_layoutc}::Stride::Index ld_bias(0);
        {r_layoutc}::Stride::Index batch_stride_bias(0);
        cutlass::half_t* bias = reinterpret_cast<cutlass::half_t*>(Bias->data);
        """
        bias_arg = """{bias, ld_bias},
        batch_stride_bias"""
    else:
        bias_param = bias_var_def = ""
        bias_arg = """{c, ldc},
        batch_stride_C"""

    text = f"""
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm.h>
      #include <cutlass/gemm/device/gemm_batched.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {{

      using namespace tvm;
      using namespace tvm::runtime;

      void _BHGEMM(NDArray A, NDArray B, {bias_param}NDArray C) {{
        // A: [Batch, M, K], B: [K, N], C: [Batch, M, N]
        CHECK_EQ(A->ndim, 3);
        int bdim = B->ndim;
        CHECK_EQ(C->ndim, 3);
        CHECK_EQ(A->shape[2], B->shape[bdim - 2]);
        int Batch = A->shape[0];
        int M = A->shape[1];
        int K = A->shape[2];
        int N = B->shape[bdim - 1];
        CHECK_EQ(C->shape[0], Batch);
        CHECK_EQ(C->shape[1], M);
        CHECK_EQ(C->shape[2], N);
        // Define the GEMM operation
        {cutlass_op_def}
        {op_name} gemm_op;
        {typec} alpha(1.0);
        {typec} beta(0.0);
        {r_layouta}::Stride::Index lda(K);
        {r_layouta}::Stride::Index batch_stride_A(M * K);
        {r_layoutb}::Stride::Index ldb(N);
        {r_layoutb}::Stride::Index batch_stride_B(bdim == 2 ? 0 : K * N);
        {r_layoutc}::Stride::Index ldc(N);
        {r_layoutc}::Stride::Index batch_stride_C(M * N);
        {bias_var_def}
        {typea}* a = reinterpret_cast<cutlass::half_t*>(A->data);
        {typeb}* b = reinterpret_cast<cutlass::half_t*>(B->data);
        {typec}* c = reinterpret_cast<cutlass::half_t*>(C->data);
        cutlass::Status status = gemm_op({{
            {{M, N, K}},     //
            {{a, lda}},      //
            batch_stride_A,   //
            {{b, ldb}},      //
            batch_stride_B,   //
            {bias_arg},      //
            {{c, ldc}},      //
            batch_stride_C,   //
            {{alpha, beta}},  //
            Batch             //
        }});
        CHECK(status == cutlass::Status::kSuccess);
      }}

      }}  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({{global_symbol}}, _BHGEMM);
      """
    print(text)
    return text

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

import tvm
from tvm import tir, relax
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import IRBuilder
from tvm.relax.transform import OperatorLegalizer
from tvm import register_func

from .cutlass_codegen import get_graph_pattern_cutlass_code

OP_PATTERN_LIST = list()
OP_PATTERN_FUNC_LIST = dict()
OP_PATTERN_VARS_LIST = dict()
OP_PATTERN_ATTR_LIST = dict()


def register_pattern(arg_spaces):
    def register(func):
        for arg in arg_spaces:
            func(*arg)
        return func

    return register


def get_value(evaluated_symbols, pattern_name):
    return [evaluated_symbols[symbol] for symbol in OP_PATTERN_VARS_LIST[pattern_name]]


@register_func("tvm.relax.cutlass.op_pattern_stitch")
def op_pattern_stitch(evaluated_symbols, evaluated_buffers, matched_pattern_names):
    attr = OP_PATTERN_ATTR_LIST[matched_pattern_names[0]]
    if matched_pattern_names[0].startswith("dense"):
        A_dense, B_dense, C_dense = evaluated_buffers[0]
        m, n, k = A_dense.shape[0], B_dense.shape[1], A_dense.shape[1]
        attr["m"] = m
        attr["n"] = n
        attr["k"] = k
    elif matched_pattern_names[0].startswith("batch_dense"):
        A_dense, B_dense, C_dense = evaluated_buffers[0]
        _, m, k = A_dense.shape
        n = B_dense.shape[-1]
        attr["m"] = m
        attr["n"] = n
        attr["k"] = k

    if len(matched_pattern_names) >= 3:
        assert len(evaluated_symbols) >= 3
        assert len(evaluated_buffers) >= 3
        if (
            matched_pattern_names[0].startswith("dense")
            and matched_pattern_names[1].startswith("bias")
            and matched_pattern_names[2].startswith("relu")
        ):
            # dense + bias + relu
            m_dense, n_dense, k_dense = get_value(evaluated_symbols[0], matched_pattern_names[0])
            m_bias, n_bias = get_value(evaluated_symbols[1], matched_pattern_names[1])
            m_relu, n_relu = get_value(evaluated_symbols[2], matched_pattern_names[2])
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            A_relu, B_relu = evaluated_buffers[2]
            if (
                m_dense == m_bias
                and n_dense == n_bias
                and m_dense == m_relu
                and n_dense == n_relu
                and C_dense == A_bias
                and C_bias == A_relu
            ):
                attr["op_type"] = "cutlass.dense_bias_relu"
                return [get_graph_pattern_cutlass_code(matched_pattern_names[:3], attr=attr), 3]
    if len(matched_pattern_names) >= 2:
        assert len(evaluated_symbols) >= 2
        assert len(evaluated_buffers) >= 2
        # dense + bias
        if matched_pattern_names[0].startswith("dense") and matched_pattern_names[1].startswith(
            "bias"
        ):
            m_dense, n_dense, k_dense = get_value(evaluated_symbols[0], matched_pattern_names[0])
            m_bias, n_bias = get_value(evaluated_symbols[1], matched_pattern_names[1])
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            if m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
                attr["op_type"] = "cutlass.dense_bias"
                return [get_graph_pattern_cutlass_code(matched_pattern_names[:2], attr=attr), 2]
        # batch_dense + batch_bias
        if matched_pattern_names[0].startswith("batch_dense") and matched_pattern_names[
            1
        ].startswith("batch_bias"):
            b_dense, m_dense, n_dense, k_dense = get_value(
                evaluated_symbols[0], matched_pattern_names[0]
            )
            b_bias, m_bias, n_bias = get_value(evaluated_symbols[1], matched_pattern_names[1])
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            if b_dense == b_bias and m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
                attr["op_type"] = "cutlass.batch_matmul_bias"
                return [get_graph_pattern_cutlass_code(matched_pattern_names[:2], attr=attr), 2]
        # # padding2d_NHWC + conv2d_NHWC
        # if (
        #     matched_pattern_names[0] in ["padding_2d_NHWC", "copy_4d"]
        #     and matched_pattern_names[1] == "conv2d_NHWC"
        # ):
        #     if matched_pattern_names[0] == "padding_2d_NHWC":
        #         (
        #             N_pad,
        #             H_pad,
        #             W_pad,
        #             C_pad,
        #             pH_pad,
        #             pW_pad,
        #             lH_pad,
        #             lW_pad,
        #             rH_pad,
        #             rW_pad,
        #         ) = get_value(evaluated_symbols[0], "padding_2d_NHWC")
        #     else:
        #         (
        #             N_pad,
        #             H_pad,
        #             W_pad,
        #             C_pad,
        #         ) = get_value(evaluated_symbols[0], "copy_4d")
        #         pH_pad = rH_pad = H_pad
        #         pW_pad = rW_pad = W_pad
        #         lH_pad = lW_pad = 0
        #     (
        #         N_conv,
        #         pH_conv,
        #         pW_conv,
        #         H_conv,
        #         W_conv,
        #         C_conv,
        #         O_conv,
        #         KH_conv,
        #         KW_conv,
        #         stride_h_conv,
        #         stride_w_conv,
        #         dilation_h_conv,
        #         dilation_w_conv,
        #     ) = get_value(evaluated_symbols[1], "conv2d_NHWC")
        #     A, A_pad = evaluated_buffers[0]
        #     A_pad_conv, B_conv, out_conv = evaluated_buffers[1]
        #     if (
        #         N_pad == N_conv
        #         and pH_pad == pH_conv
        #         and pW_pad == pW_conv
        #         and C_pad == C_conv
        #         and A_pad == A_pad_conv
        #     ):
        #         if (
        #             lH_pad == pH_pad - rH_pad
        #             and lW_pad == pW_pad - rW_pad
        #             and lH_pad + H_pad == rH_pad
        #             and lW_pad + W_pad == rW_pad
        #         ):
        #             padding = (lH_pad, lW_pad)
        #             strides = (stride_h_conv, stride_w_conv)
        #             dilation = (dilation_h_conv, dilation_w_conv)
        #             return [
        #                 get_graph_pattern_cutlass_code(
        #                     matched_pattern_names[:2],
        #                     padding=padding,
        #                     strides=strides,
        #                     dilation=dilation,
        #                 ),
        #                 2,
        #             ]
    if len(matched_pattern_names) >= 1:
        assert len(evaluated_symbols) >= 1
        assert len(evaluated_buffers) >= 1
        if matched_pattern_names[0].startswith("dense"):
            # dense
            attr["op_type"] = "cutlass.dense"
            return [get_graph_pattern_cutlass_code(matched_pattern_names[:1], attr=attr), 1]
        elif matched_pattern_names[0].startswith("batch_dense"):
            # batch_dense
            attr["op_type"] = "cutlass.batch_matmul"
            return [get_graph_pattern_cutlass_code(matched_pattern_names[:1], attr=attr), 1]
    return ["", 0]


# A_TYPE = "float16"
# B_TYPE = "float16"
# C_TYPE = "float16"


@register_func("tvm.relax.cutlass.get_op_pattern_list")
def get_op_pattern_list():
    return OP_PATTERN_LIST


@register_func("tvm.relax.cutlass.get_op_pattern_func")
def get_op_pattern_func(name):
    return OP_PATTERN_FUNC_LIST[name]


@register_func("tvm.relax.cutlass.get_op_pattern_vars")
def get_op_pattern_vars(name):
    return OP_PATTERN_VARS_LIST[name]


# register all possibile op configuration
# only profile the op configuration we use


@register_pattern([["float16", "float16", "float16"]])
def dense_row_row_row(A_TYPE, B_TYPE, C_TYPE):
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    K = tir.Var("K", "int64")
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                C = R.matmul(A, B, out_dtype=C_TYPE)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "dense_row_row_row_" + A_TYPE + "_" + B_TYPE + "_" + C_TYPE
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["matmul"]
    params = relax_mod["matmul"].params
    M = relax_mod["matmul"].buffer_map[params[0]].shape[0]
    N = relax_mod["matmul"].buffer_map[params[1]].shape[1]
    K = relax_mod["matmul"].buffer_map[params[0]].shape[1]
    OP_PATTERN_VARS_LIST[name] = [M, N, K]
    OP_PATTERN_ATTR_LIST[name] = {
        "typea": A_TYPE,
        "typeb": B_TYPE,
        "typec": C_TYPE,
        "layouta": "row",
        "layoutb": "row",
        "layoutc": "row",
    }


@register_pattern([["float16", "float16"]])
def bias_row(A_TYPE, B_TYPE):
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, N), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((1, N), B_TYPE)
                )  # pylint: disable=invalid-name
                C = R.add(A, B)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "bias_row_" + A_TYPE + "_" + B_TYPE
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["add"]
    params = relax_mod["add"].params
    M, N = relax_mod["add"].buffer_map[params[0]].shape
    OP_PATTERN_VARS_LIST[name] = [M, N]


@register_pattern([["float16", "float16"]])
def batch_bias_row(A_TYPE, B_TYPE):
    batch = tir.Var("batch", "int64")
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, N), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((1, N), B_TYPE)
                )  # pylint: disable=invalid-name
                C = R.add(A, B)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "batch_bias_row" + A_TYPE + "_" + B_TYPE
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["add"]
    params = relax_mod["add"].params
    batch, M, N = relax_mod["add"].buffer_map[params[0]].shape
    OP_PATTERN_VARS_LIST[name] = [batch, M, N]


@register_pattern([["float16"]])
def relu(A_TYPE):
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, N), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.nn.relu(A)
                R.func_ret_value(B)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "relu_" + A_TYPE
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["relu"]
    params = relax_mod["relu"].params
    M, N = relax_mod["relu"].buffer_map[params[0]].shape
    OP_PATTERN_VARS_LIST[name] = [M, N]


@register_pattern([["float16", "float16", "float16"]])
def batch_dense_row_row_row(A_TYPE, B_TYPE, C_TYPE):
    batch = tir.Var("batch", "int64")
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    K = tir.Var("K", "int64")

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                C = R.matmul(A, B, out_dtype=C_TYPE)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "batch_dense_row_row_row_" + A_TYPE + "_" + B_TYPE + "_" + C_TYPE
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["matmul"]
    params = relax_mod["matmul"].params
    batch, M, K = relax_mod["matmul"].buffer_map[params[0]].shape
    _, N = relax_mod["matmul"].buffer_map[params[1]].shape
    OP_PATTERN_VARS_LIST[name] = [batch, M, N, K]
    OP_PATTERN_ATTR_LIST[name] = {
        "typea": A_TYPE,
        "typeb": B_TYPE,
        "typec": C_TYPE,
        "layouta": "row",
        "layoutb": "row",
        "layoutc": "row",
    }


@register_pattern([["float16", "float16", "float16"]])
def batch_dense_row_row_row_2(A_TYPE, B_TYPE, C_TYPE):
    batch = tir.Var("batch", "int64")
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    K = tir.Var("K", "int64")

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((batch, K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                C = R.matmul(A, B, out_dtype=C_TYPE)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "batch_dense_2_row_row_row_" + A_TYPE + "_" + B_TYPE + "_" + C_TYPE
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["matmul"]
    params = relax_mod["matmul"].params
    batch, M, K = relax_mod["matmul"].buffer_map[params[0]].shape
    _, _, N = relax_mod["matmul"].buffer_map[params[1]].shape
    OP_PATTERN_VARS_LIST[name] = [batch, M, N, K]
    OP_PATTERN_ATTR_LIST[name] = {
        "typea": A_TYPE,
        "typeb": B_TYPE,
        "typec": C_TYPE,
        "layouta": "row",
        "layoutb": "row",
        "layoutc": "row",
    }


# @register_pattern()
# def copy_4d():
#     N = tir.Var("N", "int64")
#     H = tir.Var("H", "int64")
#     W = tir.Var("W", "int64")
#     C = tir.Var("C", "int64")

#     from tvm.script import tir as T

#     @tvm.script.ir_module
#     class Copy4D:
#         @T.prim_func
#         def main(A: T.Buffer((N, H, W, C), A_TYPE), B: T.Buffer((N, H, W, C), B_TYPE)) -> None:
#             for n, h, w, c in T.grid(N, H, W, C):
#                 with T.block("copy"):
#                     vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
#                     T.reads([A[vn, vh, vw, vc]])
#                     T.writes([B[vn, vh, vw, vc]])
#                     B[vn, vh, vw, vc] = A[vn, vh, vw, vc]

#     mod = Copy4D
#     name = "copy_4d"
#     OP_PATTERN_LIST.append(name)
#     OP_PATTERN_FUNC_LIST[name] = mod["main"]
#     OP_PATTERN_VARS_LIST[name] = [N, H, W, C]


# @register_pattern()
# def padding_2d():
#     N = tir.Var("N", "int64")
#     H = tir.Var("H", "int64")
#     W = tir.Var("W", "int64")
#     pH = tir.Var("pH", "int64")
#     pW = tir.Var("pW", "int64")
#     lH = tir.Var("lH", "int64")
#     lW = tir.Var("lW", "int64")
#     rH = tir.Var("rH", "int64")
#     rW = tir.Var("rW", "int64")
#     C = tir.Var("C", "int64")

#     from tvm.script.ir_builder import tir as T

#     with IRBuilder() as ib:
#         with I.ir_module() as frame:
#             with T.prim_func():
#                 A = T.arg("A", T.buffer_decl((N, H, W, C), A_TYPE))
#                 B = T.arg("B", T.buffer_decl((N, pH, pW, C), B_TYPE))
#                 T.func_name("main")
#                 with T.grid(N, pH, pW, C) as (n, ph, pw, c):
#                     with T.block("copy"):
#                         vn, vph, vpw, vc = T.axis.remap("SSSS", [n, ph, pw, c])
#                         T.reads([A[vn, vph - lH, vpw - lW, vc]])
#                         T.writes([B[vn, vph, vpw, vc]])
#                         T.buffer_store(
#                             B,
#                             T.if_then_else(
#                                 tvm.tir.all(lH <= vph, vph < rH, lW <= vpw, vpw < rW),
#                                 A[vn, vph - lH, vpw - lW, vc],
#                                 T.float16(0.0),
#                             ),
#                             [vn, vph, vpw, vc],
#                         )
#     mod = ib.get()
#     name = "padding_2d_NHWC"
#     OP_PATTERN_LIST.append(name)
#     OP_PATTERN_FUNC_LIST[name] = mod["main"]
#     OP_PATTERN_VARS_LIST[name] = [N, H, W, C, pH, pW, lH, lW, rH, rW]


# @register_pattern()
# def conv2d():
#     N = tir.Var("N", "int64")
#     pH = tir.Var("pH", "int64")
#     pW = tir.Var("pW", "int64")
#     H = tir.Var("H", "int64")
#     W = tir.Var("W", "int64")
#     C = tir.Var("C", "int64")
#     O = tir.Var("K", "int64")
#     KH = tir.Var("R", "int64")
#     KW = tir.Var("S", "int64")
#     StrideH = tir.Var("StrideH", "int64")
#     StrideW = tir.Var("StrideW", "int64")
#     DilateH = tir.Var("DilateH", "int64")
#     DilateW = tir.Var("DilateW", "int64")

#     from tvm.script.ir_builder import tir as T

#     with IRBuilder() as ib:
#         with I.ir_module() as frame:
#             with T.prim_func():
#                 A = T.arg("A", T.buffer_decl((N, pH, pW, C), A_TYPE))
#                 B = T.arg("B", T.buffer_decl((O, KH, KW, C), B_TYPE))
#                 out = T.arg("out", T.buffer_decl((N, H, W, O), C_TYPE))
#                 T.func_name("main")
#                 with T.grid(N, H, W, O, KH, KW, C) as (n, h, w, o, rh, rw, c):
#                     with T.block("conv"):
#                         vn, vh, vw, vo, vrh, vrw, vc = T.axis.remap(
#                             "SSSSRRR", [n, h, w, o, rh, rw, c]
#                         )
#                         T.reads(
#                             [
#                                 A[
#                                     vn,
#                                     vrh * DilateH + vh * StrideH,
#                                     vrw * DilateW + vw * StrideW,
#                                     vc,
#                                 ],
#                                 B[vo, vrh, vrw, vc],
#                             ]
#                         )
#                         T.writes([out[vn, vh, vw, vo]])
#                         with T.init():
#                             T.buffer_store(out, T.float16(0.0), [vn, vh, vw, vo])
#                         T.buffer_store(
#                             out,
#                             out[vn, vh, vw, vo]
#                             + A[vn, vrh * DilateH + vh * StrideH, vrw * DilateW + vw * StrideW, vc]
#                             * B[vo, vrh, vrw, vc],
#                             [vn, vh, vw, vo],
#                         )
#     mod = ib.get()
#     name = "conv2d_NHWC"
#     OP_PATTERN_LIST.append(name)
#     OP_PATTERN_FUNC_LIST[name] = mod["main"]
#     OP_PATTERN_VARS_LIST[name] = [N, pH, pW, H, W, C, O, KH, KW, StrideH, StrideW, DilateH, DilateW]

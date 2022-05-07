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
from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import tvm
from tvm import relax
from tvm.ir.base import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R


def test_split_preproc():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def matmul(
                A: T.Buffer[(16, 16), "float32"],
                B: T.Buffer[(16, 16), "float32"],
                C: T.Buffer[(16, 16), "float32"],
        ) -> None:
            T.func_attr({"layout_rewrite_buffers": [1]})
            B_ = T.alloc_buffer([16, 4, 4], dtype="float32")
            for i0_o, i1_o in T.grid(16, 16):
                with T.block("layout_rewrite"):
                    i0, i1 = T.axis.remap("SS", [i0_o, i1_o])
                    T.reads(B[i0, i1])
                    T.writes(B_[i1, i0 // 4, i0 % 4])
                    T.block_attr({"preproc": True})
                    B_[i1, i0 // 4, i0 % 4] = B[i0, i1]
            for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
                with T.block("matmul"):
                    vi = T.axis.spatial(16, i0 * 4 + i1)
                    vj = T.axis.spatial(16, j)
                    vk = T.axis.reduce(16, k0 * 4 + k1)
                    T.reads(A[vi, vk], B_[vj, vk // 4, vk % 4])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B_[vj, vk // 4, vk % 4]

        @R.function
        def foo(
                x: Tensor[(16, 16), "float32"], w: Tensor[(16, 16), "float32"]
        ) -> Tensor[(16, 16), "float32"]:
            gv0 = R.call_tir(matmul, (x, w), (16, 16), dtype="float32")
            return gv0

    @tvm.script.ir_module
    class OutputModule:
        @R.function
        def foo(
                x: Tensor[(16, 16), "float32"], w: Tensor[(16, 16), "float32"]
        ) -> Tensor[(16, 16), "float32"]:
            # block 0
            gv = relax.call_tir(layout_rewrite0, x, (4, 4, 4, 4), dtype="float32")
            gv0 = relax.call_tir(tir_matmul, (gv, w), (16, 16), dtype="float32")
            return gv0

        @T.prim_func
        def tir_matmul(
                A_1: T.Buffer[(4, 4, 4, 4), "float32"],
                B_1: T.Buffer[(16, 16), "float32"],
                C_1: T.Buffer[(16, 16), "float32"],
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "tir_matmul"})
            # body
            # with T.block("root")
            for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
                with T.block("matmul"):
                    vi = T.axis.spatial(16, i0 * 4 + i1)
                    vj = T.axis.spatial(16, j)
                    vk = T.axis.reduce(16, k0 * 4 + k1)
                    T.reads(A_1[vi // 4, vk // 4, vi % 4, vk % 4], B_1[vk, vj])
                    T.writes(C_1[vi, vj])
                    with T.init():
                        C_1[vi, vj] = T.float32(0)
                    C_1[vi, vj] = C_1[vi, vj] + A_1[vi // 4, vk // 4, vi % 4, vk % 4] * B_1[vk, vj]

        @T.prim_func
        def layout_rewrite0(
                src_1: T.Buffer[(16, 16), "float32"], tgt_1: T.Buffer[(4, 4, 4, 4), "float32"]
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "layout_rewrite0", "tir.noalias": True})
            # body
            # with T.block("root")
            for i0v, i1v in T.grid(16, 16):
                with T.block("layout_rewrite"):
                    i0, i1 = T.axis.remap("SS", [i0v, i1v])
                    T.reads(src_1[i0, i1])
                    T.writes(tgt_1[i0 // 4, i1 // 4, i0 % 4, i1 % 4])
                    tgt_1[i0 // 4, i1 // 4, i0 % 4, i1 % 4] = src_1[i0, i1]

    mod = Before
    after = relax.transform.SplitPreProc()(mod)
    print(R.parser.astext(after))
    # assert_structural_equal(new_mod, OutputModule, map_free_vars=True)


if __name__ == "__main__":
    test_split_preproc()

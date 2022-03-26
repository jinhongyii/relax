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

import tvm.script
from tvm.script import tir as T, relax as R


def test_annotate_opkind_outewisefusable():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int32")
            n = T.var("int32")
            k = T.var("int32")
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def foo(x: Tensor[(m, n), "float32"], w: Tensor[(n, k), "float32"]) -> Tensor:
            gv0 = R.call_tir( tir_matmul, (x, w), (m, k), dtype="float32")
            return gv0

    mod = InputModule
    new_mod =relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_matmul"].attrs["op_pattern"] == 4

def test_annotate_opkind_reduce():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def sum(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "elemwise"})
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16,))

            for i, j in T.grid(16, 16):
                with T.block("matmul"):
                    vi, vj = T.axis.remap("SR", [i, j])
                    with T.init():
                        B[vi] = 0.
                    B[vi] += A[vi, vj]

        @R.function
        def foo(x: Tensor[(16, 16), "float32"]) -> Tensor:
            gv0 = R.call_tir(sum, (x), (16,), dtype="float32")
            return gv0
    mod = InputModule
    new_mod =relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["sum"].attrs["op_pattern"] == 3

def test_annotate_opkind_ewise():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def elemwise(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "elemwise"})
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16, 16))

            for i, j in T.grid(16, 16):
                with T.block("matmul"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]+1.0

        @R.function
        def foo(x: Tensor[(16, 16), "float32"]) -> Tensor:
            gv0 = R.call_tir(elemwise, (x), (16, 16), dtype="float32")
            return gv0

    mod = InputModule
    new_mod =relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["elemwise"].attrs["op_pattern"] == 0

def test_annotate_opkind_broadcast():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def broadcast(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "elemwise"})
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16, 16, 16, 16))

            for i0, j0, i1, j1 in T.grid(16, 16, 16, 16):
                with T.block("matmul"):
                    vi0, vj0, vi1, vj1 = T.axis.remap("SSSS", [i0, j0, i1, j1])
                    B[vi0, vj0, vi1, vj1] = A[vj0, vj1]

        @R.function
        def foo(x: Tensor[(16, 16), "float32"]) -> Tensor:
            gv0 = R.call_tir(broadcast, (x, ), (16, 16, 16, 16), dtype="float32")
            return gv0

    mod = InputModule
    new_mod =relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["broadcast"].attrs["op_pattern"] == 1

def test_annotate_opkind_injective():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def injective(x: T.handle, y: T.handle) -> None:
            T.func_attr({"global_symbol": "elemwise"})
            A = T.match_buffer(x, (4, 4, 4, 4))
            B = T.match_buffer(y, (16, 16))

            for i, j in T.grid(16, 16):
                with T.block("matmul"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi//4, vj//4, vi%4, vj%4]

        @R.function
        def foo(x: Tensor[(4, 4, 4, 4), "float32"]) -> Tensor:
            gv0 = R.call_tir(injective, (x, ), (16, 16), dtype="float32")
            return gv0

    mod = InputModule
    new_mod =relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["injective"].attrs["op_pattern"] == 2

def test_annotate_op_kind_bias_add():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def tir_bias_add(rxplaceholder_2: T.Buffer[(1, 1000), "float32"], rxplaceholder_3: T.Buffer[(1000,), "float32"], T_add_1: T.Buffer[(1, 1000), "float32"]) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "tir_bias_add", "tir.noalias": True})
            # body
            # with T.block("root")
            for i0, i1 in T.grid(1, 1000):
                with T.block("T_add"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder_2[ax0, ax1], rxplaceholder_3[ax1])
                    T.writes(T_add_1[ax0, ax1])
                    T_add_1[ax0, ax1] = rxplaceholder_2[ax0, ax1] + rxplaceholder_3[ax1]

        @R.function
        def foo(x: Tensor[(1, 1000), "float32"], y: Tensor[(1000, ), "float32"]) -> Tensor:
            gv0 = R.call_tir(tir_bias_add, (x, y), (1, 1000), dtype="float32")
            return gv0

    mod = InputModule
    new_mod =relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_bias_add"].attrs["op_pattern"] == 1

if __name__ == "__main__":
    pytest.main([__file__])

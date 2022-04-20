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
import numpy as np
import tvm.script
from tvm.script import tir as T, relax as R

def gen_mod(mod, name, binding):
    """Select relax function with name, rename to main and and bind constant.

    Parameters
    ----------
    mod: IRModule
        The input module

    name: str
        The name of relax function to preserve and rename to main

    binding: Dict[str, array]
        The const parameter bindings
    """
    funcs = {}
    binding = {k: tvm.nd.array(v) for k, v in binding.items()}

    for k, v in mod.functions.items():
        if isinstance(v, tvm.relax.Function):
            if k.name_hint == name:
                # rename to main
                gv = tvm.ir.GlobalVar("main")
                funcs[gv] = tvm.relax.Function(v.params, v.body, v.ret_type, gv)
        else:
            funcs[k] = v
    mod = tvm.IRModule(funcs)
    return relax.transform.BindParams("main", binding)(mod)

def test_annotate_simple():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16, 16))
            C = T.match_buffer(z, (16, 16))
            for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
                with T.block("matmul"):
                    vi = T.axis.S(16, i0 * 4 + i1)
                    vj = T.axis.S(16, j)
                    vk = T.axis.R(16, k0 * 4 + k1)
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def foo(
                x: Tensor[(16, 16), "float32"], w: Tensor[(16, 16), "float32"]
        ) -> Tensor[(16, 16), "float32"]:
            gv0 = R.call_tir(tir_matmul, (x, w), (16, 16), dtype="float32")
            return gv0

    @tvm.script.ir_module
    class After:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"layout_rewrite_buffers": [1]})
            A = T.match_buffer(x, (16, 16))
            B = T.match_buffer(y, (16, 16))
            C = T.match_buffer(z, (16, 16))
            for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
                with T.block("matmul"):
                    vi = T.axis.S(16, i0 * 4 + i1)
                    vj = T.axis.S(16, j)
                    vk = T.axis.R(16, k0 * 4 + k1)
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def foo(
                x: Tensor[(16, 16), "float32"], w: Tensor[(16, 16), "float32"]
        ) -> Tensor[(16, 16), "float32"]:
            gv0 = R.call_tir(tir_matmul, (x, w), (16, 16), dtype="float32")
            return gv0

    w_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)

    before = gen_mod(Before, "foo", {"w": w_np})
    after = relax.transform.AnnotateLayoutRewriteBuffers()(before)
    expected = gen_mod(After, "foo", {"w":w_np})
    assert_structural_equal(after, expected)


if __name__ == "__main__":
    test_annotate_simple()

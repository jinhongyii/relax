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
import tvm.testing
from tvm import relax, relay
from tvm.ir.base import assert_structural_equal
import numpy as np

import tvm.script
from tvm.script import tir as T, relax as R


def check(mod):
    x_tvm = tvm.nd.array(np.ones((16, 16), "float32"))
    mod = relax.transform.BindParams({"x": x_tvm})(mod)
    mod = relax.transform.FoldConstant()(mod)

    ret = None

    def fvisit(e):
        if isinstance(e, relax.SeqExpr):
            assert isinstance(e.body, relay.Constant)
            nonlocal ret
            ret = e.body.data

    relax.analysis.post_order_visit(mod["main"], fvisit)
    tvm.testing.assert_allclose(ret.numpy(), x_tvm.numpy())

def test_one_fold():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def identity(A: T.Buffer[(16, 16), "float32"], B: T.Buffer[(16, 16), "float32"]) -> None:
            for i, j in T.grid(16, 16):
                with T.block("identity"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @R.function
        def main(x: Tensor[(16, 16), "float32"]):
            lv0 = relax.call_tir(identity, (x,), (16, 16), dtype="float32")
            return lv0

    check(InputModule)


def test_two_fold():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def identity(A: T.Buffer[(16, 16), "float32"], B: T.Buffer[(16, 16), "float32"]) -> None:
            for i, j in T.grid(16, 16):
                with T.block("identity"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @R.function
        def main(x: Tensor[(16, 16), "float32"]):
            lv0 = relax.call_tir(identity, (x,), (16, 16), dtype="float32")
            lv1 = relax.call_tir(identity, (lv0,), (16, 16), dtype="float32")
            return lv1

    check(InputModule)

def test_dataflow_fold():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def identity(A: T.Buffer[(16, 16), "float32"], B: T.Buffer[(16, 16), "float32"]) -> None:
            for i, j in T.grid(16, 16):
                with T.block("identity"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @R.function
        def main(x: Tensor[(16, 16), "float32"]):
            with R.dataflow():
                gv0 = relax.call_tir(identity, (x,), (16, 16), dtype="float32")
                R.output(gv0)
            return gv0

    check(InputModule)

def test_dynamic_shape():
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def identity(a: T.handle, b: T.handle, m: T.int32, n: T.int32) -> None:
            A = T.match_buffer(a, (m, n))
            B = T.match_buffer(b, (m, n))
            for i, j in T.grid(m, n):
                with T.block("identity"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @R.function
        def main(x: Tensor[(m, n), "float32"]):
            lv0 = relax.call_tir(identity, (x, ), (m, n), (m, n), dtype="float32")
            return lv0
    check(InputModule)

if __name__ == "__main__":
    test_one_fold()
    test_two_fold()
    test_dataflow_fold()
    # test_dynamic_shape()


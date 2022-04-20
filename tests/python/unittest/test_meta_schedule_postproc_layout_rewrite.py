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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

import tvm
from tvm import tir
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.postproc import LayoutRewrite
from tvm.script import tir as T
from tvm.target import Target


def _target() -> Target:
    return Target("cuda", host="llvm")


def _create_context(mod, target) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        postprocs=[
            LayoutRewrite(),
        ],
        task_name="test",
    )
    for rule in ctx.postprocs:
        rule.initialize_with_tune_context(ctx)
    return ctx

@tvm.script.ir_module
class Before:
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

@tvm.script.ir_module
class After:
    @T.prim_func
    def tir_matmul(A_1: T.Buffer[(16, 16), "float32"], B_2: T.Buffer[(16, 16), "float32"], C_1: T.Buffer[(16, 16), "float32"]) -> None:
        # function attr dict
        T.func_attr({"layout_rewrite_buffers": [1]})
        # body
        # with T.block("root")
        B_3 = T.alloc_buffer([16, 4, 4], dtype="float32")
        for i0_o, i1_o in T.grid(16, 16):
            with T.block("layout_rewrite"):
                i0, i1 = T.axis.remap("SS", [i0_o, i1_o])
                T.reads(B_2[i0, i1])
                T.writes(B_3[i1, i0 // 4, i0 % 4])
                T.block_attr({"preproc":True})
                B_3[i1, i0 // 4, i0 % 4] = B_2[i0, i1]
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with T.block("matmul"):
                vi = T.axis.spatial(16, i0 * 4 + i1)
                vj = T.axis.spatial(16, j)
                vk = T.axis.reduce(16, k0 * 4 + k1)
                T.reads(A_1[vi, vk], B_3[vj, vk // 4, vk % 4])
                T.writes(C_1[vi, vj])
                with T.init():
                    C_1[vi, vj] = T.float32(0)
                C_1[vi, vj] = C_1[vi, vj] + A_1[vi, vk] * B_3[vj, vk // 4, vk % 4]

def test_layout_rewrite():
    mod = Before
    target = _target()
    ctx = _create_context(mod, target)
    sch = tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, After)


if __name__ == "__main__":
    test_layout_rewrite()

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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys

import pytest

import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


def packed_index_map_func(m, n):
    return m // 16, n // 16, m % 16, n % 16


@T.prim_func
def two_elementwise(A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks
# fmt: on


@T.prim_func
def two_elementwise_transformed_with_preproc(
    A_2: T.Buffer[(128, 128), "float32"], C_1: T.Buffer[(128, 128), "float32"]
) -> None:
    # body
    # with T.block("root")
    B_1 = T.alloc_buffer([128, 128], dtype="float32")
    A_3 = T.alloc_buffer([8, 8, 16, 16], dtype="float32")
    for m_o, n_o in T.grid(128, 128):
        with T.block("layout_rewrite"):
            m, n = T.axis.remap("SS", [m_o, n_o])
            T.reads(A_2[m, n])
            T.writes(A_3[m // 16, n // 16, m % 16, n % 16])
            T.block_attr({"preproc": True})
            A_3[m // 16, n // 16, m % 16, n % 16] = A_2[m, n]
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A_3[vi // 16, vj // 16, vi % 16, vj % 16])
            T.writes(B_1[vi, vj])
            B_1[vi, vj] = A_3[vi // 16, vj // 16, vi % 16, vj % 16] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B_1[vi, vj])
            T.writes(C_1[vi, vj])
            C_1[vi, vj] = B_1[vi, vj] + T.float32(1)


def test_two_elementwise_transform_input_buffer():
    sch = tir.Schedule(two_elementwise, debug_mask="all")
    block = sch.get_block("B")
    sch.transform_layout_with_preproc(block, 0, "read", packed_index_map_func)
    tvm.ir.assert_structural_equal(two_elementwise_transformed_with_preproc, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)

def test_two_elementwise_transform_intermediate_buffer():
    pass

def test_two_elementwise_transform_output_buffer():
    pass

if __name__ == "__main__":
    test_two_elementwise_transform_input_buffer()

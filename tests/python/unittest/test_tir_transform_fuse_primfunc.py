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
# pylint: disable=missing-function-docstring, missing-module-docstring

import tvm
from tvm.script import tir as T


@T.prim_func
def matmul(
    A: T.Buffer[(16, 32), "float32"],
    B: T.Buffer[(16, 32), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    for i, j, k in T.grid(16, 16, 32):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def add(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"global_symbol": "add", "tir.noalias": True})
    for i, j in T.grid(16, 16):
        with T.block("add_one"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + B[vi, vj]


@T.prim_func
def matmul_add(
    A: T.Buffer[(16, 32), "float32"],
    B: T.Buffer[(16, 32), "float32"],
    D: T.Buffer[(16, 16), "float32"],
    E: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"global_symbol": "fused_matmul_add", "tir.noalias": True})
    C = T.alloc_buffer([16, 16], dtype="float32")

    for i, j, k in T.grid(16, 16, 32):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    for i, j in T.grid(16, 16):
        with T.block("add_one"):
            vi, vj = T.axis.remap("SS", [i, j])
            E[vi, vj] = C[vi, vj] + D[vi, vj]


@T.prim_func
def matmul_add_2(
    A: T.Buffer[(16, 32), "float32"],
    B: T.Buffer[(16, 32), "float32"],
    D: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"global_symbol": "fused_matmul_add", "tir.noalias": True})
    C = T.alloc_buffer([16, 16], dtype="float32")

    for i, j, k in T.grid(16, 16, 32):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    for i, j in T.grid(16, 16):
        with T.block("add_one"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = C[vi, vj] + C[vi, vj]


@T.prim_func
def matmul_block_name_compute(
    A: T.Buffer[(16, 32), "float32"],
    B: T.Buffer[(16, 32), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    for i, j, k in T.grid(16, 16, 32):
        with T.block("compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def add_block_name_compute(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"global_symbol": "add", "tir.noalias": True})
    for i, j in T.grid(16, 16):
        with T.block("compute"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + B[vi, vj]


@T.prim_func
def matmul_add_name_deduplication(
    A: T.Buffer[(16, 32), "float32"],
    B: T.Buffer[(16, 32), "float32"],
    D: T.Buffer[(16, 16), "float32"],
    E: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"global_symbol": "fused_matmul_add", "tir.noalias": True})
    C = T.alloc_buffer([16, 16], dtype="float32")

    for i, j, k in T.grid(16, 16, 32):
        with T.block("compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    for i, j in T.grid(16, 16):
        with T.block("compute_1"):
            vi, vj = T.axis.remap("SS", [i, j])
            E[vi, vj] = C[vi, vj] + D[vi, vj]


def test_simple():
    f = tvm.tir.fuse_primfuncs([matmul, add], {add.params[0]: matmul.params[2]})
    tvm.ir.assert_structural_equal(f, matmul_add)


def test_multiple():
    f = tvm.tir.fuse_primfuncs(
        [matmul, add], {add.params[0]: matmul.params[2], add.params[1]: matmul.params[2]}
    )
    tvm.ir.assert_structural_equal(f, matmul_add_2)


def test_name_deduplication():
    f = tvm.tir.fuse_primfuncs(
        [matmul_block_name_compute, add_block_name_compute],
        {add_block_name_compute.params[0]: matmul_block_name_compute.params[2]},
    )
    tvm.ir.assert_structural_equal(f, matmul_add_name_deduplication)


if __name__ == "__main__":
    test_simple()
    test_multiple()
    test_name_deduplication()

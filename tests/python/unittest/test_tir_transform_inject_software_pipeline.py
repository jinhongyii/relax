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
import pytest
import sys
import numpy as np

import tvm
import tvm.testing
import tvm.tir.tensor_intrin.cuda
from tvm import tir, te, TVMError
from tvm.script import tir as T
from tvm.meta_schedule.testing import te_workload
from tvm.testing.tir import mma_schedule
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_16x16_A_DYN_INTRIN,
    LDMATRIX_16x16_B_DYN_INTRIN,
    MMA_f16f16f32_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
)


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.InjectSoftwarePipeline()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


def _check_error(func):
    mod = tvm.IRModule.from_expr(func)
    with pytest.raises(ValueError):
        tvm.tir.transform.InjectSoftwarePipeline()(mod)


@T.prim_func
def trivial_pipeline(A: T.Buffer[(16, 1), "float32"], C: T.Buffer[(16, 1), "float32"]):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0, 1, annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]}
        ):
            with T.block():
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


@T.prim_func
def transformed_trivial_pipeline(
    A: T.Buffer[(16, 1), "float32"], C: T.Buffer[(16, 1), "float32"]
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with T.block():
            T.reads(A[tx, 0])
            T.writes(C[tx, 0])
            B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads(A[tx, 0])
                T.writes(B[0, tx, 0])
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with T.block():
                T.reads()
                T.writes()
                T.evaluate(0)
            with T.block():
                T.reads(B[0, tx, 0])
                T.writes(C[tx, 0])
                C[tx, 0] = B[0, tx, 0] + T.float32(1)


def gen_simple_compute(num_stages):
    @T.prim_func
    def simple_compute(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                16,
                annotations={
                    "software_pipeline_stage": [0, num_stages],
                    "software_pipeline_order": [0, 1],
                },
            ):
                with T.block("compute"):
                    T.reads(A[tx, i])
                    T.writes(C[tx, i])
                    B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                    with T.block():
                        T.reads(A[tx, i])
                        T.writes(B[tx, 0])
                        B[tx, 0] = A[tx, i] * T.float32(2)
                    with T.block():
                        T.reads(B[tx, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = B[tx, 0] + T.float32(1)

    return simple_compute


@T.prim_func
def transformed_simple_compute(
    A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16]])
            T.writes([C[tx, 0:16]])
            B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0]])
                T.writes([B[0, tx, 0]])
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with T.block():
                T.reads([A[tx, 1:16], B[0:2, tx, 0]])
                T.writes([B[0:2, tx, 0], C[tx, 0:15]])
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1]])
                        T.writes([B[(i + 1) % 2, tx, 0]])
                        B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
                    with T.block():
                        T.reads([B[i % 2, tx, 0]])
                        T.writes([C[tx, i]])
                        C[tx, i] = B[i % 2, tx, 0] + T.float32(1)
            with T.block():
                T.reads([B[1, tx, 0]])
                T.writes([C[tx, 15]])
                C[tx, 15] = B[1, tx, 0] + T.float32(1)


@T.prim_func
def three_stage_compute(A: T.Buffer[(16, 16), "float32"], D: T.Buffer[(16, 16), "float32"]):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1, 2],
                "software_pipeline_order": [0, 1, 2],
            },
        ):
            with T.block("compute"):
                T.reads(A[tx, i])
                T.writes(D[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                C = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, 0])
                    C[tx, 0] = B[tx, 0] + T.float32(2)
                with T.block():
                    T.reads(C[tx, 0])
                    T.writes(D[tx, i])
                    D[tx, i] = C[tx, 0] + T.float32(1)


@T.prim_func
def transformed_three_stage_compute(
    A: T.Buffer[(16, 16), "float32"], D: T.Buffer[(16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with T.block():
            T.reads(A[tx, 0:16])
            T.writes(D[tx, 0:16])
            B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            C = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads(A[tx, 0:2], B[0:2, tx, 0])
                T.writes(B[0:2, tx, 0], C[0:2, tx, 0])
                for i in T.unroll(2):
                    with T.block():
                        T.reads(A[tx, i])
                        T.writes(B[0:2, tx, 0])
                        B[i, tx, 0] = A[tx, i] * T.float32(2)
                    with T.block():
                        T.where(1 <= i)
                        T.reads(B[0:2, tx, 0])
                        T.writes(C[0:2, tx, 0])
                        C[(i + 1) % 2, tx, 0] = B[(i + 1) % 2, tx, 0] + T.float32(2)
            with T.block():
                T.reads(A[tx, 2:16], B[0:2, tx, 0], C[0:2, tx, 0])
                T.writes(B[0:2, tx, 0], C[0:2, tx, 0], D[tx, 0:14])
                for i in T.serial(14):
                    with T.block():
                        T.reads(A[tx, i + 2])
                        T.writes(B[0:2, tx, 0])
                        B[i % 2, tx, 0] = A[tx, i + 2] * T.float32(2)
                    with T.block():
                        T.reads(B[0:2, tx, 0])
                        T.writes(C[0:2, tx, 0])
                        C[(i + 1) % 2, tx, 0] = B[(i + 1) % 2, tx, 0] + T.float32(2)
                    with T.block():
                        T.reads(C[0:2, tx, 0])
                        T.writes(D[tx, i])
                        D[tx, i] = C[i % 2, tx, 0] + T.float32(1)
            with T.block():
                T.reads(B[0:2, tx, 0], C[0:2, tx, 0])
                T.writes(C[0:2, tx, 0], D[tx, 14:16])
                for i in T.unroll(2):
                    with T.block():
                        T.where(i < 1)
                        T.reads(B[0:2, tx, 0])
                        T.writes(C[0:2, tx, 0])
                        C[(i + 1) % 2, tx, 0] = B[(i + 1) % 2, tx, 0] + T.float32(2)
                    with T.block():
                        T.reads(C[0:2, tx, 0])
                        T.writes(D[tx, i + 14])
                        D[tx, i + 14] = C[i, tx, 0] + T.float32(1)


@T.prim_func
def dag_interleaving(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 0, 0, 0, 1],
                "software_pipeline_order": [0, 2, 1, 3, 4],
            },
        ):
            with T.block():
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                AS = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                BS = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                AL = T.alloc_buffer((1, 1), dtype="float32", scope="local")
                BL = T.alloc_buffer((1, 1), dtype="float32", scope="local")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(AS[tx, 0])
                    AS[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(AS[tx, 0])
                    T.writes(AL[0, 0])
                    AL[0, 0] = AS[tx, 0]
                with T.block():
                    T.reads(B[tx, i])
                    T.writes(BS[tx, 0])
                    BS[tx, 0] = B[tx, i] + T.float32(2)
                with T.block():
                    T.reads(BS[tx, 0])
                    T.writes(BL[0, 0])
                    BL[0, 0] = BS[tx, 0]
                with T.block():
                    T.reads(AL[0, 0], BL[0, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = AL[0, 0] * BL[0, 0]


@T.prim_func
def transformed_dag_interleaving(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    for tx in T.thread_binding(16, thread="threadIdx.x"):
        with T.block():
            T.reads(A[tx, 0:16], B[tx, 0:16])
            T.writes(C[tx, 0:16])
            AS = T.alloc_buffer([16, 1], dtype="float32", scope="shared")
            BS = T.alloc_buffer([16, 1], dtype="float32", scope="shared")
            AL = T.alloc_buffer([2, 1, 1], dtype="float32", scope="local")
            BL = T.alloc_buffer([2, 1, 1], dtype="float32", scope="local")
            with T.block():
                T.reads(A[tx, 0], B[tx, 0], AS[tx, 0], BS[tx, 0])
                T.writes(AS[tx, 0], BS[tx, 0], AL[0, 0, 0], BL[0, 0, 0])
                with T.block():
                    T.reads(A[tx, 0])
                    T.writes(AS[tx, 0])
                    AS[tx, 0] = A[tx, 0] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(BS[tx, 0])
                    BS[tx, 0] = B[tx, 0] + T.float32(2)
                with T.block():
                    T.reads(AS[tx, 0])
                    T.writes(AL[0, 0, 0])
                    AL[0, 0, 0] = AS[tx, 0]
                with T.block():
                    T.reads(BS[tx, 0])
                    T.writes(BL[0, 0, 0])
                    BL[0, 0, 0] = BS[tx, 0]
            with T.block():
                T.reads(
                    A[tx, 1:16], B[tx, 1:16], AS[tx, 0], BS[tx, 0], AL[0:2, 0, 0], BL[0:2, 0, 0]
                )
                T.writes(AS[tx, 0], BS[tx, 0], AL[0:2, 0, 0], BL[0:2, 0, 0], C[tx, 0:15])
                for i in T.serial(15):
                    with T.block():
                        T.reads(A[tx, i + 1])
                        T.writes(AS[tx, 0])
                        AS[tx, 0] = A[tx, i + 1] * T.float32(2)
                    with T.block():
                        T.reads(B[tx, i + 1])
                        T.writes(BS[tx, 0])
                        BS[tx, 0] = B[tx, i + 1] + T.float32(2)
                    with T.block():
                        T.reads(AS[tx, 0])
                        T.writes(AL[(i + 1) % 2, 0, 0])
                        AL[(i + 1) % 2, 0, 0] = AS[tx, 0]
                    with T.block():
                        T.reads(BS[tx, 0])
                        T.writes(BL[(i + 1) % 2, 0, 0])
                        BL[(i + 1) % 2, 0, 0] = BS[tx, 0]
                    with T.block():
                        T.reads(AL[i % 2, 0, 0], BL[i % 2, 0, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = AL[i % 2, 0, 0] * BL[i % 2, 0, 0]
            with T.block():
                T.reads(AL[1, 0, 0], BL[1, 0, 0])
                T.writes(C[tx, 15])
                C[tx, 15] = AL[1, 0, 0] * BL[1, 0, 0]


@T.prim_func
def nested_pipeline_simple(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1, 1, 1],
                "software_pipeline_order": [0, 1, 2, 3],
            },
        ):
            with T.block():
                T.reads(A[tx, i, 0:16])
                T.writes(C[tx, i, 0:16])
                A_shared = T.alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A[tx, i, j])
                        T.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with T.block():
                        T.reads(A_shared[tx, 0, j])
                        T.writes(C[tx, i, j])
                        B = T.alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with T.block():
                            T.reads(A_shared[tx, i, j])
                            T.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_shared[tx, 0, j] * T.float32(2)
                        with T.block():
                            T.reads(B[tx, i, 0])
                            T.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@T.prim_func
def transformed_nested_pipeline_simple(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16, 0:16]])
            T.writes([C[tx, 0:16, 0:16]])
            A_shared = T.alloc_buffer([2, 16, 1, 16], dtype="float32", scope="shared")
            B = T.alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0, 0:16]])
                T.writes([A_shared[0, tx, 0, 0:16]])
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads([A[tx, 0, j]])
                        T.writes([A_shared[0, tx, 0, j]])
                        A_shared[0, tx, 0, j] = A[tx, 0, j]
            with T.block():
                T.reads([A[tx, 1:16, 0:16], A_shared[0:2, tx, 0:15, 0:16], B[0:2, tx, 0:15, 0]])
                T.writes([A_shared[0:2, tx, 0, 0:16], B[0:2, tx, 0:15, 0], C[tx, 0:15, 0:16]])
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1, 0:16]])
                        T.writes([A_shared[(i + 1) % 2, tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A[tx, i + 1, j]])
                                T.writes([A_shared[(i + 1) % 2, tx, 0, j]])
                                A_shared[(i + 1) % 2, tx, 0, j] = A[tx, i + 1, j]
                    with T.block():
                        T.reads([A_shared[i % 2, tx, i, 0]])
                        T.writes([B[0, tx, i, 0]])
                        B[0, tx, i, 0] = A_shared[i % 2, tx, 0, 0] * T.float32(2)
                    with T.block():
                        T.reads([A_shared[i % 2, tx, i, 1:16], B[0:2, tx, i, 0]])
                        T.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with T.block():
                                T.reads([A_shared[i % 2, tx, i, j + 1]])
                                T.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_shared[
                                    i % 2, tx, 0, j + 1
                                ] * T.float32(2)
                            with T.block():
                                T.reads([B[j % 2, tx, i, 0]])
                                T.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with T.block():
                        T.reads([B[1, tx, i, 0]])
                        T.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with T.block():
                T.reads([A_shared[1, tx, 15, 0:16], B[0:2, tx, 15, 0]])
                T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with T.block():
                    T.reads([A_shared[1, tx, 15, 0]])
                    T.writes([B[0, tx, 15, 0]])
                    B[0, tx, 15, 0] = A_shared[1, tx, 0, 0] * T.float32(2)
                with T.block():
                    T.reads([A_shared[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with T.block():
                            T.reads([A_shared[1, tx, 15, j + 1]])
                            T.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_shared[1, tx, 0, j + 1] * T.float32(2)
                        with T.block():
                            T.reads([B[j % 2, tx, 15, 0]])
                            T.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with T.block():
                    T.reads([B[1, tx, 15, 0]])
                    T.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@T.prim_func
def nested_pipeline_prefetch_inner(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 0, 1, 1],
                "software_pipeline_order": [0, 2, 1, 3],
            },
        ):
            with T.block():
                T.reads(A[tx, i, 0:16])
                T.writes(C[tx, i, 0:16])
                A_shared = T.alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A[tx, i, j])
                        T.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with T.block():
                        T.reads(A_shared[tx, 0, j])
                        T.writes(C[tx, i, j])
                        B = T.alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with T.block():
                            T.reads(A_shared[tx, i, j])
                            T.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_shared[tx, 0, j] * T.float32(2)
                        with T.block():
                            T.reads(B[tx, i, 0])
                            T.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@T.prim_func
def transformed_nested_pipeline_prefetch_inner(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16, 0:16]])
            T.writes([C[tx, 0:16, 0:16]])
            A_shared = T.alloc_buffer([2, 16, 1, 16], dtype="float32", scope="shared")
            B = T.alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0, 0:16], A_shared[0, tx, 0, 0]])
                T.writes([A_shared[0, tx, 0, 0:16], B[0, tx, 0, 0]])
                with T.block():
                    T.reads([A[tx, 0, 0:16]])
                    T.writes([A_shared[0, tx, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A[tx, 0, j]])
                            T.writes([A_shared[0, tx, 0, j]])
                            A_shared[0, tx, 0, j] = A[tx, 0, j]
                with T.block():
                    T.reads([A_shared[0, tx, 0, 0]])
                    T.writes([B[0, tx, 0, 0]])
                    B[0, tx, 0, 0] = A_shared[0, tx, 0, 0] * T.float32(2)
            with T.block():
                T.reads([A[tx, 1:16, 0:16], A_shared[0:2, tx, 0:16, 0:16], B[0:2, tx, 0:15, 0]])
                T.writes([A_shared[0:2, tx, 0, 0:16], B[0:2, tx, 0:16, 0], C[tx, 0:15, 0:16]])
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1, 0:16]])
                        T.writes([A_shared[(i + 1) % 2, tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A[tx, i + 1, j]])
                                T.writes([A_shared[(i + 1) % 2, tx, 0, j]])
                                A_shared[(i + 1) % 2, tx, 0, j] = A[tx, i + 1, j]
                    with T.block():
                        T.reads([A_shared[i % 2, tx, i, 1:16], B[0:2, tx, i, 0]])
                        T.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with T.block():
                                T.reads([A_shared[i % 2, tx, i, j + 1]])
                                T.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_shared[
                                    i % 2, tx, 0, j + 1
                                ] * T.float32(2)
                            with T.block():
                                T.reads([B[j % 2, tx, i, 0]])
                                T.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with T.block():
                        T.reads([A_shared[(i + 1) % 2, tx, i + 1, 0]])
                        T.writes([B[0, tx, i + 1, 0]])
                        B[0, tx, i + 1, 0] = A_shared[(i + 1) % 2, tx, 0, 0] * T.float32(2)
                    with T.block():
                        T.reads([B[1, tx, i, 0]])
                        T.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with T.block():
                T.reads([A_shared[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with T.block():
                    T.reads([A_shared[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with T.block():
                            T.reads([A_shared[1, tx, 15, j + 1]])
                            T.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_shared[1, tx, 0, j + 1] * T.float32(2)
                        with T.block():
                            T.reads([B[j % 2, tx, 15, 0]])
                            T.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with T.block():
                    T.reads([B[1, tx, 15, 0]])
                    T.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@T.prim_func
def nested_pipeline_interleaving(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 0, 0, 1, 1],
                "software_pipeline_order": [0, 2, 3, 1, 4],
            },
        ):
            with T.block():
                T.reads(A[tx, i, 0:16])
                T.writes(C[tx, i, 0:16])
                A_shared = T.alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                A_local = T.alloc_buffer((1, 1, 16), dtype="float32", scope="local")
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A[tx, i, j])
                        T.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A_shared[tx, 0, j])
                        T.writes(A_local[0, 0, j])
                        A_local[0, 0, j] = A_shared[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with T.block():
                        T.reads(A_local[0, 0, j])
                        T.writes(C[tx, i, j])
                        B = T.alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with T.block():
                            T.reads(A_local[tx, i, j])
                            T.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_local[0, 0, j] * T.float32(2)
                        with T.block():
                            T.reads(B[tx, i, 0])
                            T.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@T.prim_func
def transformed_nested_pipeline_interleaving(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16, 0:16]])
            T.writes([C[tx, 0:16, 0:16]])
            A_shared = T.alloc_buffer([16, 1, 16], dtype="float32", scope="shared")
            A_local = T.alloc_buffer([1, 1, 16], dtype="float32", scope="local")
            B = T.alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0, 0:16], A_shared[tx, 0, 0:16], A_local[tx, 0, 0]])
                T.writes([A_shared[tx, 0, 0:16], A_local[0, 0, 0:16], B[0, tx, 0, 0]])
                with T.block():
                    T.reads([A[tx, 0, 0:16]])
                    T.writes([A_shared[tx, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A[tx, 0, j]])
                            T.writes([A_shared[tx, 0, j]])
                            A_shared[tx, 0, j] = A[tx, 0, j]
                with T.block():
                    T.reads([A_shared[tx, 0, 0:16]])
                    T.writes([A_local[0, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A_shared[tx, 0, j]])
                            T.writes([A_local[0, 0, j]])
                            A_local[0, 0, j] = A_shared[tx, 0, j]
                with T.block():
                    T.reads([A_local[tx, 0, 0]])
                    T.writes([B[0, tx, 0, 0]])
                    B[0, tx, 0, 0] = A_local[0, 0, 0] * T.float32(2)
            with T.block():
                T.reads(
                    [
                        A[tx, 1:16, 0:16],
                        A_local[tx, 0:16, 0:16],
                        B[0:2, tx, 0:15, 0],
                        A_shared[tx, 0, 0:16],
                    ]
                )
                T.writes(
                    [
                        A_shared[tx, 0, 0:16],
                        B[0:2, tx, 0:16, 0],
                        C[tx, 0:15, 0:16],
                        A_local[0, 0, 0:16],
                    ]
                )
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1, 0:16]])
                        T.writes([A_shared[tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A[tx, i + 1, j]])
                                T.writes([A_shared[tx, 0, j]])
                                A_shared[tx, 0, j] = A[tx, i + 1, j]
                    with T.block():
                        T.reads([A_local[tx, i, 1:16], B[0:2, tx, i, 0]])
                        T.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with T.block():
                                T.reads([A_local[tx, i, j + 1]])
                                T.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_local[0, 0, j + 1] * T.float32(2)
                            with T.block():
                                T.reads([B[j % 2, tx, i, 0]])
                                T.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with T.block():
                        T.reads([A_shared[tx, 0, 0:16]])
                        T.writes([A_local[0, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A_shared[tx, 0, j]])
                                T.writes([A_local[0, 0, j]])
                                A_local[0, 0, j] = A_shared[tx, i + 1, j]
                    with T.block():
                        T.reads([A_local[tx, i + 1, 0]])
                        T.writes([B[0, tx, i + 1, 0]])
                        B[0, tx, i + 1, 0] = A_local[0, 0, 0] * T.float32(2)
                    with T.block():
                        T.reads([B[1, tx, i, 0]])
                        T.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with T.block():
                T.reads([A_local[tx, 15, 1:16], B[0:2, tx, 15, 0]])
                T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with T.block():
                    T.reads([A_local[tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with T.block():
                            T.reads([A_local[tx, 15, j + 1]])
                            T.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_local[0, 0, j + 1] * T.float32(2)
                        with T.block():
                            T.reads([B[j % 2, tx, 15, 0]])
                            T.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with T.block():
                    T.reads([B[1, tx, 15, 0]])
                    T.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@T.prim_func
def nested_pipeline_double_buffer(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 0, 0, 1, 1],
                "software_pipeline_order": [0, 2, 3, 1, 4],
            },
        ):
            with T.block():
                T.reads(A[tx, i, 0:16])
                T.writes(C[tx, i, 0:16])
                A_shared = T.alloc_buffer((16, 1, 16), dtype="float32", scope="shared")
                A_local = T.alloc_buffer((1, 1, 16), dtype="float32", scope="local")
                for j in T.serial(0, 16):
                    with T.block():
                        T.reads(A[tx, i, j])
                        T.writes(A_shared[tx, 0, j])
                        A_shared[tx, 0, j] = A[tx, i, j]
                for j in T.serial(0, 16):
                    with T.block():
                        T.block_attr({"double_buffer_scope": 0})
                        T.reads(A_shared[tx, 0, j])
                        T.writes(A_local[0, 0, j])
                        A_local[0, 0, j] = A_shared[tx, i, j]
                for j in T.serial(
                    0,
                    16,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1],
                    },
                ):
                    with T.block():
                        T.reads(A_local[0, 0, j])
                        T.writes(C[tx, i, j])
                        B = T.alloc_buffer((16, 1, 1), dtype="float32", scope="shared")
                        with T.block():
                            T.reads(A_local[tx, i, j])
                            T.writes(B[tx, i, 0])
                            B[tx, i, 0] = A_local[0, 0, j] * T.float32(2)
                        with T.block():
                            T.reads(B[tx, i, 0])
                            T.writes(C[tx, i, j])
                            C[tx, i, j] = B[tx, i, 0] + T.float32(1)


@T.prim_func
def transformed_nested_pipeline_double_buffer(
    A: T.Buffer[(16, 16, 16), "float32"], C: T.Buffer[(16, 16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16, 0:16]])
            T.writes([C[tx, 0:16, 0:16]])
            A_shared = T.alloc_buffer([16, 1, 16], dtype="float32", scope="shared")
            A_local = T.alloc_buffer([2, 1, 1, 16], dtype="float32", scope="local")
            B = T.alloc_buffer([2, 16, 1, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0, 0:16], A_shared[tx, 0, 0:16], A_local[0, tx, 0, 0]])
                T.writes([A_shared[tx, 0, 0:16], A_local[0, 0, 0, 0:16], B[0, tx, 0, 0]])
                with T.block():
                    T.reads([A[tx, 0, 0:16]])
                    T.writes([A_shared[tx, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A[tx, 0, j]])
                            T.writes([A_shared[tx, 0, j]])
                            A_shared[tx, 0, j] = A[tx, 0, j]
                with T.block():
                    T.reads([A_shared[tx, 0, 0:16]])
                    T.writes([A_local[0, 0, 0, 0:16]])
                    for j in T.serial(0, 16):
                        with T.block():
                            T.reads([A_shared[tx, 0, j]])
                            T.writes([A_local[0, 0, 0, j]])
                            T.block_attr({"double_buffer_scope": 0})
                            A_local[0, 0, 0, j] = A_shared[tx, 0, j]
                with T.block():
                    T.reads([A_local[0, tx, 0, 0]])
                    T.writes([B[0, tx, 0, 0]])
                    B[0, tx, 0, 0] = A_local[0, 0, 0, 0] * T.float32(2)
            with T.block():
                T.reads(
                    [
                        A[tx, 1:16, 0:16],
                        A_local[0:2, tx, 0:16, 0:16],
                        B[0:2, tx, 0:15, 0],
                        A_shared[tx, 0, 0:16],
                    ]
                )
                T.writes(
                    [
                        A_shared[tx, 0, 0:16],
                        B[0:2, tx, 0:16, 0],
                        C[tx, 0:15, 0:16],
                        A_local[0:2, 0, 0, 0:16],
                    ]
                )
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1, 0:16]])
                        T.writes([A_shared[tx, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A[tx, i + 1, j]])
                                T.writes([A_shared[tx, 0, j]])
                                A_shared[tx, 0, j] = A[tx, i + 1, j]
                    with T.block():
                        T.reads([A_local[i % 2, tx, i, 1:16], B[0:2, tx, i, 0]])
                        T.writes([B[0:2, tx, i, 0], C[tx, i, 0:15]])
                        for j in T.serial(0, 15):
                            with T.block():
                                T.reads([A_local[i % 2, tx, i, j + 1]])
                                T.writes([B[(j + 1) % 2, tx, i, 0]])
                                B[(j + 1) % 2, tx, i, 0] = A_local[i % 2, 0, 0, j + 1] * T.float32(
                                    2
                                )
                            with T.block():
                                T.reads([B[j % 2, tx, i, 0]])
                                T.writes([C[tx, i, j]])
                                C[tx, i, j] = B[j % 2, tx, i, 0] + T.float32(1)
                    with T.block():
                        T.reads([A_shared[tx, 0, 0:16]])
                        T.writes([A_local[(i + 1) % 2, 0, 0, 0:16]])
                        for j in T.serial(0, 16):
                            with T.block():
                                T.reads([A_shared[tx, 0, j]])
                                T.writes([A_local[(i + 1) % 2, 0, 0, j]])
                                T.block_attr({"double_buffer_scope": 0})
                                A_local[(i + 1) % 2, 0, 0, j] = A_shared[tx, i + 1, j]
                    with T.block():
                        T.reads([A_local[(i + 1) % 2, tx, i + 1, 0]])
                        T.writes([B[0, tx, i + 1, 0]])
                        B[0, tx, i + 1, 0] = A_local[(i + 1) % 2, 0, 0, 0] * T.float32(2)
                    with T.block():
                        T.reads([B[1, tx, i, 0]])
                        T.writes([C[tx, i, 15]])
                        C[tx, i, 15] = B[1, tx, i, 0] + T.float32(1)
            with T.block():
                T.reads([A_local[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:16]])
                with T.block():
                    T.reads([A_local[1, tx, 15, 1:16], B[0:2, tx, 15, 0]])
                    T.writes([B[0:2, tx, 15, 0], C[tx, 15, 0:15]])
                    for j in T.serial(0, 15):
                        with T.block():
                            T.reads([A_local[1, tx, 15, j + 1]])
                            T.writes([B[(j + 1) % 2, tx, 15, 0]])
                            B[(j + 1) % 2, tx, 15, 0] = A_local[1, 0, 0, j + 1] * T.float32(2)
                        with T.block():
                            T.reads([B[j % 2, tx, 15, 0]])
                            T.writes([C[tx, 15, j]])
                            C[tx, 15, j] = B[j % 2, tx, 15, 0] + T.float32(1)
                with T.block():
                    T.reads([B[1, tx, 15, 0]])
                    T.writes([C[tx, 15, 15]])
                    C[tx, 15, 15] = B[1, tx, 15, 0] + T.float32(1)


@T.prim_func
def simple_compute_incorrect_reorder(
    A: T.Buffer[(16, 16), "float32"], D: T.Buffer[(16, 16), "float32"]
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1, 1],
                "software_pipeline_order": [0, 2, 1],
            },
        ):
            with T.block():
                T.reads(A[tx, i])
                T.writes(D[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                C = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, 0])
                    C[tx, 0] = B[tx, 0] + T.float32(2)
                with T.block():
                    T.reads(C[tx, 0])
                    T.writes(D[tx, i])
                    D[tx, i] = C[tx, 0] + T.float32(1)


@T.prim_func
def simple_compute_conflicting_order(
    A: T.Buffer[(16, 16), "float32"], D: T.Buffer[(16, 16), "float32"]
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, 1, 1],
                "software_pipeline_order": [0, 1, 1],
            },
        ):
            with T.block():
                T.reads(A[tx, i])
                T.writes(D[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                C = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, 0])
                    C[tx, 0] = B[tx, 0] + T.float32(2)
                with T.block():
                    T.reads(C[tx, 0])
                    T.writes(D[tx, i])
                    D[tx, i] = C[tx, 0] + T.float32(1)


@T.prim_func
def simple_compute_missing_annotation(
    A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]
):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(0, 16, annotations={"software_pipeline_stage": [0, 1]}):
            with T.block():
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


def test_simple_compute():
    _check(gen_simple_compute(1), transformed_simple_compute)


def test_trivial_pipeline():
    _check(trivial_pipeline, transformed_trivial_pipeline)


def test_three_stage_compute():
    _check(three_stage_compute, transformed_three_stage_compute)


def test_dag_interleaving():
    _check(dag_interleaving, transformed_dag_interleaving)


def test_nest_pipeline_simple():
    _check(nested_pipeline_simple, transformed_nested_pipeline_simple)


def test_nest_pipeline_prefetch_inner():
    _check(nested_pipeline_prefetch_inner, transformed_nested_pipeline_prefetch_inner)


def test_nest_pipeline_interleaving():
    _check(nested_pipeline_interleaving, transformed_nested_pipeline_interleaving)


def test_nest_pipeline_double_buffer():
    _check(nested_pipeline_double_buffer, transformed_nested_pipeline_double_buffer)


def test_error_reorder():
    _check_error(simple_compute_incorrect_reorder)


def test_error_conflicting_order():
    _check_error(simple_compute_conflicting_order)


def test_error_missing_annotation():
    _check_error(simple_compute_missing_annotation)


def test_simple_compute_async():
    mod = tvm.IRModule.from_expr(gen_simple_compute(1))
    sch = tvm.tir.Schedule(mod)

    _, loop = sch.get_loops(sch.get_block("compute"))
    sch.annotate(loop, ann_key="software_pipeline_async_stages", ann_val=[0])
    mod = tvm.tir.transform.InjectSoftwarePipeline()(sch.mod)

    @T.prim_func
    def ref(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]) -> None:
        for tx in T.thread_binding(16, thread="threadIdx.x"):
            with T.block():
                T.reads(A[tx, 0 : 16])
                T.writes(C[tx, 0 : 16])
                B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, 0])
                    T.writes(B[0 % 2, tx, 0])
                    with T.attr(0, "async_scope", 1):
                        B[0 % 2, tx, 0] = A[tx, 0] * T.float32(2)
                    T.evaluate(T.async_commit_stage(0, dtype=""))
                with T.block():
                    T.reads(A[tx, 1 : 16], B[0 : 2, tx, 0])
                    T.writes(B[0 : 2, tx, 0], C[tx, 0 : 15])
                    for i in T.serial(15):
                        with T.block():
                            T.where(i + 1 < 16)
                            T.reads(A[tx, i + 1])
                            T.writes(B[(i + 1) % 2, tx, 0])
                            with T.attr(0, "async_scope", 1):
                                B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
                            T.evaluate(T.async_commit_stage(0, dtype=""))
                        with T.block():
                            T.where(i + 1 - 1 < 16)
                            T.reads(B[(i - 1 + 1) % 2, tx, 0])
                            T.writes(C[tx, i - 1 + 1])
                            T.evaluate(T.async_wait_stage(0, 1, dtype=""))
                            C[tx, i - 1 + 1] = B[(i - 1 + 1) % 2, tx, 0] + T.float32(1)
                with T.block():
                    T.reads(B[15 % 2, tx, 0])
                    T.writes(C[tx, 15])
                    T.evaluate(T.async_wait_stage(0, 0, dtype=""))
                    C[tx, 15] = B[15 % 2, tx, 0] + T.float32(1)

    tvm.ir.assert_structural_equal(mod["main"], ref, True)

    mod = tvm.IRModule.from_expr(gen_simple_compute(3))
    sch = tvm.tir.Schedule(mod)

    _, loop = sch.get_loops(sch.get_block("compute"))
    sch.annotate(loop, ann_key="software_pipeline_async_stages", ann_val=[0])
    mod = tvm.tir.transform.InjectSoftwarePipeline()(sch.mod)

    @T.prim_func
    def ref(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]) -> None:
        for tx in T.thread_binding(16, thread="threadIdx.x"):
            with T.block():
                T.reads(A[tx, 0 : 16])
                T.writes(C[tx, 0 : 16])
                B = T.alloc_buffer([4, 16, 1], dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, 0 : 3])
                    T.writes(B[0 : 4, tx, 0])
                    for i in T.unroll(3):
                        with T.block():
                            T.where(i < 16)
                            T.reads(A[tx, i])
                            T.writes(B[0 : 4, tx, 0])
                            with T.attr(0, "async_scope", 1):
                                B[i % 4, tx, 0] = A[tx, i] * T.float32(2)
                            T.evaluate(T.async_commit_stage(0, dtype=""))
                with T.block():
                    T.reads(A[tx, 3 : 16], B[0 : 4, tx, 0])
                    T.writes(B[0 : 4, tx, 0], C[tx, 0 : 13])
                    for i in T.serial(13):
                        with T.block():
                            T.where(i + 3 < 16)
                            T.reads(A[tx, i + 3])
                            T.writes(B[0 : 4, tx, 0])
                            with T.attr(0, "async_scope", 1):
                                B[(i + 3) % 4, tx, 0] = A[tx, i + 3] * T.float32(2)
                            T.evaluate(T.async_commit_stage(0, dtype=""))
                        with T.block():
                            T.where(i + 3 - 3 < 16)
                            T.reads(B[0 : 4, tx, 0])
                            T.writes(C[tx, i - 3 + 3])
                            T.evaluate(T.async_wait_stage(0, 3, dtype=""))
                            C[tx, i - 3 + 3] = B[(i - 3 + 3) % 4, tx, 0] + T.float32(1)
                with T.block():
                    T.reads(B[0 : 4, tx, 0])
                    T.writes(C[tx, 13 : 16])
                    for i in T.unroll(3):
                        with T.block():
                            T.where(i + 16 - 3 < 16)
                            T.reads(B[0 : 4, tx, 0])
                            T.writes(C[tx, i - 3 + 16])
                            T.evaluate(T.async_wait_stage(0, 2 - i, dtype=""))
                            C[tx, i - 3 + 16] = B[(i - 3 + 16) % 4, tx, 0] + T.float32(1)


    tvm.ir.assert_structural_equal(mod["main"], ref, True)


def test_three_stage_compute_two_stage_async():
    mod = tvm.IRModule.from_expr(three_stage_compute)
    sch = tvm.tir.Schedule(mod)

    _, loop = sch.get_loops(sch.get_block("compute"))
    sch.annotate(loop, ann_key="software_pipeline_async_stages", ann_val=[0, 1])

    mod = tvm.tir.transform.InjectSoftwarePipeline()(sch.mod)

    @T.prim_func
    def ref(A: T.Buffer[(16, 16), "float32"], D: T.Buffer[(16, 16), "float32"]) -> None:
        for tx in T.thread_binding(16, thread="threadIdx.x"):
            with T.block():
                T.reads(A[tx, 0 : 16])
                T.writes(D[tx, 0 : 16])
                B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
                C = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, 0 : 2], B[0 : 2, tx, 0])
                    T.writes(B[0 : 2, tx, 0], C[0 : 2, tx, 0])
                    for i in T.unroll(2):
                        with T.block():
                            T.where(i < 16)
                            T.reads(A[tx, i])
                            T.writes(B[0 : 2, tx, 0])
                            with T.attr(0, "async_scope", 1):
                                B[i % 2, tx, 0] = A[tx, i] * T.float32(2)
                            T.evaluate(T.async_commit_stage(0, dtype=""))
                        with T.block():
                            T.where(1 <= i and i - 1 < 16)
                            T.reads(B[0 : 2, tx, 0])
                            T.writes(C[0 : 2, tx, 0])
                            T.evaluate(T.async_wait_stage(0, 1, dtype=""))
                            with T.attr(0, "async_scope", 1):
                                C[(i - 1) % 2, tx, 0] = B[(i - 1) % 2, tx, 0] + T.float32(2)
                            T.evaluate(T.async_commit_stage(1, dtype=""))
                with T.block():
                    T.reads(A[tx, 2 : 16], B[0 : 2, tx, 0], C[0 : 2, tx, 0])
                    T.writes(B[0 : 2, tx, 0], C[0 : 2, tx, 0], D[tx, 0 : 14])
                    for i in T.serial(14):
                        with T.block():
                            T.where(i + 2 < 16)
                            T.reads(A[tx, i + 2])
                            T.writes(B[0 : 2, tx, 0])
                            with T.attr(0, "async_scope", 1):
                                B[(i + 2) % 2, tx, 0] = A[tx, i + 2] * T.float32(2)
                            T.evaluate(T.async_commit_stage(0, dtype=""))
                        with T.block():
                            T.where(i + 2 - 1 < 16)
                            T.reads(B[0 : 2, tx, 0])
                            T.writes(C[0 : 2, tx, 0])
                            T.evaluate(T.async_wait_stage(0, 1, dtype=""))
                            with T.attr(0, "async_scope", 1):
                                C[(i - 1 + 2) % 2, tx, 0] = B[(i - 1 + 2) % 2, tx, 0] + T.float32(2)
                            T.evaluate(T.async_commit_stage(1, dtype=""))
                        with T.block():
                            T.where(i + 2 - 2 < 16)
                            T.reads(C[0 : 2, tx, 0])
                            T.writes(D[tx, i - 2 + 2])
                            T.evaluate(T.async_wait_stage(1, 1, dtype=""))
                            D[tx, i - 2 + 2] = C[(i - 2 + 2) % 2, tx, 0] + T.float32(1)
                with T.block():
                    T.reads(B[0 : 2, tx, 0], C[0 : 2, tx, 0])
                    T.writes(C[0 : 2, tx, 0], D[tx, 14 : 16])
                    for i in T.unroll(2):
                        with T.block():
                            T.where(i + 16 - 1 < 16)
                            T.reads(B[0 : 2, tx, 0])
                            T.writes(C[0 : 2, tx, 0])
                            T.evaluate(T.async_wait_stage(0, 0 - i, dtype=""))
                            with T.attr(0, "async_scope", 1):
                                C[(i - 1 + 16) % 2, tx, 0] = B[(i - 1 + 16) % 2, tx, 0] + T.float32(2)
                            T.evaluate(T.async_commit_stage(1, dtype=""))
                        with T.block():
                            T.where(i + 16 - 2 < 16)
                            T.reads(C[0 : 2, tx, 0])
                            T.writes(D[tx, i - 2 + 16])
                            if i + 16 - 1 < 16:
                                T.evaluate(T.async_wait_stage(1, 1, dtype=""))
                            else:
                                T.evaluate(T.async_wait_stage(1, 0, dtype=""))
                            D[tx, i - 2 + 16] = C[(i - 2 + 16) % 2, tx, 0] + T.float32(1)

    tvm.ir.assert_structural_equal(mod["main"], ref, True)


N = K = M = 4096


def get_mma_schedule():
    i_factors, j_factors, k_factors = [1, 32, 1, 4, 2], [16, 2, 4, 1, 2], [128, 2, 1]

    def index_map(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
        )

    workload = te.create_prim_func(te_workload.matmul_fp16(N, M, K))

    return mma_schedule(
        workload,
        16,
        "float16",
        False,
        i_factors,
        j_factors,
        k_factors,
        index_map,
        index_map,
        index_map,
        LDMATRIX_16x16_A_DYN_INTRIN,
        LDMATRIX_16x16_B_DYN_INTRIN,
        MMA_f16f16f32_INTRIN,
        MMA_fill_16x16_f32_INTRIN,
        MMA_store_16x16_f32_global_INTRIN,
        "shared.dyn",
    )


def build_and_run(sch):
    if True or tvm.testing.is_ampere_or_newer():
        with tvm.transform.PassContext(config={"tir.use_ptx_async_copy": 1}):
            f = tvm.build(sch.mod["main"], target="cuda")

        dev = tvm.device("cuda", 0)
        a_np = np.random.uniform(size=(N, K)).astype("float16")
        b_np = np.random.uniform(size=(K, M)).astype("float16")
        c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)
        # print(f.imported_modules[0].get_source())
        print("ok")

        # evaluator = f.time_evaluator(f.entry_name, dev, number=500)
        # gflops = (N * M * K) * 2 / 1e9
        # time_ms = evaluator(a, b, c).mean * 1e3
        # print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


@tvm.testing.requires_cuda
def test_pipelined_mma_gemm_simple():
    sch = get_mma_schedule()

    k0 = sch.get_loops(sch.get_block("C_o_update"))[3]

    sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
    sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
    sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0, 1])

    seq = tvm.transform.Sequential(
        [
            tvm.tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tvm.tir.transform.ConvertBlocksToOpaque(),
            tvm.tir.transform.UnifyThreadBinding(),
            tvm.tir.transform.LowerMatchBuffer(),
            tvm.tir.transform.InjectSoftwarePipeline(),
        ]
    )
    mod = seq(sch.mod)

    pipeline = mod["main"].body.block.body.body.body.body.body.block.body[1].block.body
    prologue, body, epilogue = pipeline

    assert "tir.async_commit_stage(0)" == str(prologue.block.body.body[1].block.body[-1]).rstrip()
    assert "async_wait_stage" not in str(prologue)  # No need to wait in the prologue
    assert "tir.async_commit_stage(0)" == str(body.block.body.body[1].block.body[1]).rstrip()
    assert "tir.async_wait_stage(0, 3)" == str(body.block.body.body[2].block.body[0]).rstrip()
    assert (
        "tir.async_wait_stage(0, (2 - i2_0_0))"
        == str(epilogue.block.body.body.block.body[0]).rstrip()
    )

    # build_and_run(sch)


@tvm.testing.requires_cuda
def test_nested_pipeline_mma_gemm_ideal_annotation():
    sch = get_mma_schedule()

    k0 = sch.get_loops(sch.get_block("C_o_update"))[3]
    k1 = sch.get_loops(sch.get_block("C_o_update"))[4]

    sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 2, 3, 3])
    sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 3, 2, 4])
    sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0, 1])

    sch.annotate(k1, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
    sch.annotate(k1, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

    seq = tvm.transform.Sequential(
        [
            tvm.tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tvm.tir.transform.ConvertBlocksToOpaque(),
            tvm.tir.transform.UnifyThreadBinding(),
            tvm.tir.transform.LowerMatchBuffer(),
            tvm.tir.transform.InjectSoftwarePipeline(),
        ]
    )
    mod = seq(sch.mod)

    pipeline = mod["main"].body.block.body.body.body.body.body.block.body[1].block.body
    prologue, body, epilogue = pipeline

    assert "tir.async_commit_stage(0)" == str(prologue.block.body.body[1].block.body[-1]).rstrip()
    assert "tir.async_wait_stage(0, 2)" == str(prologue.block.body.body[2].block.body[0]).rstrip()
    assert "tir.async_commit_stage(0)" == str(body.block.body.body[1].block.body[-1]).rstrip()
    assert "tir.async_wait_stage(0, 2)" == str(body.block.body.body[2].block.body[0]).rstrip()
    assert (
        "tir.async_wait_stage(0, (1 - i2_0_0))"
        == str(epilogue.block.body.body[0].block.body[0]).rstrip()
    )

    # build_and_run(sch)


@tvm.testing.requires_cuda
def test_nested_pipeline_mma_gemm_bad_annotation():
    sch = get_mma_schedule()

    k0 = sch.get_loops(sch.get_block("C_o_update"))[3]
    k1 = sch.get_loops(sch.get_block("C_o_update"))[4]

    # This puts ldmatrix, the consumer of async copy, in the same stage as async copy
    # So we need to put wait_stage(0) before ldmatrix, to force all async copies to
    # complete immediately.
    sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, 3, 3])
    sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 3, 2, 4])
    sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0, 1])

    sch.annotate(k1, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
    sch.annotate(k1, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

    seq = tvm.transform.Sequential(
        [
            tvm.tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tvm.tir.transform.ConvertBlocksToOpaque(),
            tvm.tir.transform.UnifyThreadBinding(),
            tvm.tir.transform.LowerMatchBuffer(),
            tvm.tir.transform.InjectSoftwarePipeline(),
        ]
    )
    mod = seq(sch.mod)

    pipeline = mod["main"].body.block.body.body.body.body.body.block.body[1].block.body
    body = pipeline[1]

    assert "tir.async_wait_stage(0, 0)" == str(body.block.body.body[2].block.body[0]).rstrip()

    # build_and_run(sch)


if __name__ == "__main__":
    # tvm.testing.main()
    test_simple_compute_async()
    test_three_stage_compute_two_stage_async()
    test_pipelined_mma_gemm_simple()
    test_nested_pipeline_mma_gemm_ideal_annotation()
    test_nested_pipeline_mma_gemm_bad_annotation()

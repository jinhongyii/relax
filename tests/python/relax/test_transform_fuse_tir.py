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
import sys
import tvm
from tvm import topi, te
from tvm import relax as rx
from tvm.script import tir as T, relax as R


def test_simple():
    def before():
        bb = rx.BlockBuilder()
        x1 = rx.Var("x1", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("fused_exp_squeeze", [x1]):
            with bb.dataflow():
                lv1 = bb.emit_te(topi.exp, x1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_squeeze")
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(rx.Call(func_gv, [x]))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    def fused_exp_squeeze(x):
        exp = topi.exp(x)
        squeeze = topi.squeeze(exp)
        return squeeze

    def expected():
        bb = rx.BlockBuilder()
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_exp_squeeze, x)
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    before = before()
    after = rx.transform.FuseTIR()(before)
    expected = expected()
    tvm.ir.assert_structural_equal(after, expected)


def test_two_subfunction():
    def before():
        bb = rx.BlockBuilder()
        x1 = rx.Var("x1", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("fused_exp_squeeze", [x1]):
            with bb.dataflow():
                lv1 = bb.emit_te(topi.exp, x1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_squeeze")
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(rx.Call(func_gv, [x]))
                lv2 = bb.emit(rx.Call(func_gv, [lv]))
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    def fused_exp_squeeze(x):
        exp = topi.exp(x)
        squeeze = topi.squeeze(exp)
        return squeeze

    def expected():
        bb = rx.BlockBuilder()
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_exp_squeeze, x)
                lv2 = bb.emit_te(fused_exp_squeeze, lv)
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    before = before()
    after = rx.transform.FuseTIR()(before)
    expected = expected()
    tvm.ir.assert_structural_equal(after, expected)


def test_fuse_same_primfunc():
    def before():
        bb = rx.BlockBuilder()
        x1 = rx.Var("x1", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("fused_exp_exp_squeeze", [x1]):
            with bb.dataflow():
                lv1 = bb.emit_te(topi.exp, x1)
                lv2 = bb.emit_te(topi.exp, lv1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv2))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_exp_squeeze")
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(rx.Call(func_gv, [x]))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    def fused_exp_exp_squeeze(x):
        exp = topi.exp(x)
        exp = topi.exp(exp)
        squeeze = topi.squeeze(exp)
        return squeeze

    def expected():
        bb = rx.BlockBuilder()
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_exp_exp_squeeze, x)
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    before = before()
    after = rx.transform.FuseTIR()(before)
    expected = expected()
    tvm.ir.assert_structural_equal(after, expected)


def test_fuse_with_tuple_as_param():
    pass


def test_fuse_with_tuple_as_intermediate_var():
    pass


def test_fuse_with_call_tir_in_main():
    def before():
        bb = rx.BlockBuilder()
        x1 = rx.Var("x1", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("fused_exp_squeeze", [x1]):
            with bb.dataflow():
                lv1 = bb.emit_te(topi.exp, x1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_squeeze")
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(rx.Call(func_gv, [x]))
                lv2 = bb.emit_te(topi.add, lv, rx.const(1, "float32"))
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    def fused_exp_squeeze(x):
        exp = topi.exp(x)
        squeeze = topi.squeeze(exp)
        return squeeze

    def expected():
        bb = rx.BlockBuilder()
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_exp_squeeze, x)
                lv2 = bb.emit_te(topi.add, lv, rx.const(1, "float32"))
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    before = before()
    after = rx.transform.FuseTIR()(before)
    expected = expected()
    tvm.ir.assert_structural_equal(after, expected)


def test_fuse_with_const_in_argument():
    def before():
        bb = rx.BlockBuilder()
        x1 = rx.Var("x1", [10, 20], rx.DynTensorType(2, "float32"))
        x2 = rx.Var("x2", [], rx.DynTensorType(0, "float32"))
        with bb.function("fused_add_exp_squeeze", [x1, x2]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x1, x2)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_add_exp_squeeze")
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(rx.Call(func_gv, [x, rx.const(1, "float32")]))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    def fused_add_exp_squeeze(x, y):
        add = topi.add(x, y)
        exp = topi.exp(add)
        squeeze = topi.squeeze(exp)
        return squeeze

    def expected():
        bb = rx.BlockBuilder()
        x = rx.Var("x", [10, 20], rx.DynTensorType(2, "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_add_exp_squeeze, x, rx.const(1, "float32"))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    before = before()
    after = rx.transform.FuseTIR()(before)
    expected = expected()
    tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

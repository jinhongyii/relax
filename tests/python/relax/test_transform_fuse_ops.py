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
from __future__ import annotations # must import to defer parsing of annotations

import tvm.script
from tvm import relax
from tvm.script import tir as T, relax as R
from tvm.relay import testing
from tvm.relax.testing import relay_translator


def test_fuse_simple():
    """Simple testcase."""

    @tvm.script.ir_module
    class before:
        @R.function
        def main(x: Tensor[(m, n), "float32"]):
            with R.dataflow():
                lv0: Tensor[(m, n), "float32"] = R.call_tir((m, n), f1, (x, x))
                gv0: Tensor[(m, n), "float32"] = R.call_tir((m, n), f2, (lv0, x))
                R.output(gv0)
            return gv0

        @T.prim_func
        def f1(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "f1", "tir.noalias": True, "op_pattern": 0})
            m = T.var("int32")
            n = T.var("int32")
            A = T.match_buffer(a, (m, n))
            B = T.match_buffer(b, (m, n))
            C = T.match_buffer(c, (m, n))
            with T.block():
                T.evaluate(1)

        @T.prim_func
        def f2(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "f2", "tir.noalias": True, "op_pattern": 0})
            m = T.var("int32")
            n = T.var("int32")
            A = T.match_buffer(a, (m, n))
            B = T.match_buffer(b, (m, n))
            C = T.match_buffer(c, (m, n))
            with T.block():
                T.evaluate(2)

    mod = relax.transform.FuseOps()(before)
    print(R.parser.astext(mod))


def test_fuse_resnet():
    relay_mod, _ = testing.resnet.get_workload(num_layers=50, batch_size=1, dtype="float32")
    mod = relay_translator.from_relay(relay_mod["main"])

    mod = relax.transform.AnnotateOpKind()(mod)
    mod = relax.transform.FuseOps()(mod)
    print(R.parser.astext(mod["main"]))


if __name__ == "__main__":
    # test_fuse_simple()
    test_fuse_resnet()
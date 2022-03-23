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
from __future__ import annotations
import tvm
from tvm.script import tir as T, relax as R
from tvm import relax, relay
import numpy as np
from tvm.tir import Schedule
from tvm.ir.module import IRModule
from tvm.target.target import Target
import tempfile
from typing import List
from tvm.meta_schedule import ReplayTraceConfig, tune_tir
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord
from tvm.meta_schedule.integration import extract_task_from_relax
from tvm import transform
import time
import pytest

from tvm.relay import testing
from tvm.relax.testing import relay_translator, nn


class DummyDatabase(PyDatabase):
    def __init__(self):
        super().__init__()
        self.records = []
        self.workload_reg = []

    def has_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return True
        return False

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.records.append(record)

    def commit_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return workload
        workload = Workload(mod)
        self.workload_reg.append(workload)
        return workload

    def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
        return list(
            filter(
                lambda x: x.workload == workload,
                sorted(self.records, key=lambda x: sum(x.run_secs) / len(x.run_secs)),
            )
        )[: int(top_k)]

    def __len__(self) -> int:
        return len(self.records)

    def print_results(self) -> None:
        print("\n".join([str(r) for r in self.records]))


@pytest.mark.parametrize("dev", ["cpu"])
def test_class_irmodule(dev: str):
    relay_mod, params = testing.resnet.get_workload(num_layers=18, batch_size=1, dtype="float32")
    relay_mod = relay.transform.SimplifyInference()(relay_mod)
    main_func = relay_mod["main"]
    bind_main_func = relay.build_module.bind_params_by_name(main_func, params)

    # translate the ResNet model from Relay to Relax
    relax_mod = relay_translator.from_relay(bind_main_func)
    relax_mod = relax.transform.AnnotateOpKind()(relax_mod)
    relax_mod = relax.transform.FuseOps()(relax_mod)


    mod = relax_mod
    assert isinstance(mod, tvm.IRModule)

    if dev == "cpu":
        target = Target("llvm --num-cores=16")
        dev = tvm.cpu()
    else:
        target = Target("nvidia/geforce-rtx-3070")
        dev = tvm.cuda()

    database = DummyDatabase()
    tasks = extract_task_from_relax(mod, target=target)
    for task in tasks:
        print(f"Extracted task: {task.task_name}, {task.target}")
        with tempfile.TemporaryDirectory() as work_dir:
            sch: Schedule = tune_tir(
                mod=task.mod,
                target=target,
                config=ReplayTraceConfig(
                    num_trials_per_iter=32,
                    num_trials_total=32,
                ),
                work_dir=work_dir,
                database=database,
            )

    with transform.PassContext(opt_level=3):
        ex0 = relax.vm.build(mod, target)

    with transform.PassContext(opt_level=3):
        mod = relax.transform.MetaScheduleApplyHistoryBest(database, target)(mod)
        mod = relax.transform.LayoutRewrite()(mod)
        mod = relax.transform.FoldConstant()(mod)
        ex1 = relax.vm.build(mod, target)

    vm0 = relax.VirtualMachine(ex0, dev)
    vm1 = relax.VirtualMachine(ex1, dev)
    shape = (1, 3, 224, 224)
    data = tvm.nd.array(np.random.rand(*shape).astype(np.float32))

    # Measure the performance w/o tuning log
    tic = time.time()
    vm0["main"](data)
    toc = time.time()
    e0 = toc - tic

    # Measure the performance w/ tuning log
    tic = time.time()
    vm1["main"](data)
    toc = time.time()
    e1 = toc - tic

    print(f"w/o tuning: {e0}")
    print(f"w/  tuning: {e1}")


if __name__ == "__main__":
    pytest.main([__file__])

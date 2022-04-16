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
import os

from typing import Dict

import tvm
import tvm.testing
import tvm.relay.testing
import tvm.meta_schedule as ms

from tvm import tir, relay, relax, runtime
from tvm import transform
from tvm.ir.module import IRModule
from tvm.meta_schedule import tune_relax, EvolutionarySearchConfig
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relax.testing import relay_translator
from tvm.target.target import Target

import argparse
import logging
import numpy as np


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--device",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
    )
    args.add_argument(
        "--tune-model",
        type=int,
        required=True,
    )
    args.add_argument(
        "--layout",
        type=str,
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    if parsed.target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
        parsed.alloc_repeat = 3
    else:
        parsed.alloc_repeat = 1
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=180,
    )
    parsed.rpc_workers = parsed.rpc_config.count_num_servers(allow_missing=False)
    parsed.device = tvm.cpu() if parsed.device == "cpu" else tvm.cuda()
    parsed.tune_model = False if parsed.tune_model == 0 else True
    return parsed


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


def apply_opt_before_tuning(relay_mod: IRModule, params: Dict[str, runtime.NDArray]):
    with transform.PassContext(opt_level=3):
        main_func = relay_mod["main"]
        bind_main_func = relay.build_module.bind_params_by_name(main_func, params)
        relay_mod = IRModule.from_expr(bind_main_func)
        relay_mod = relay.transform.SimplifyInference()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)

        relax_mod = relay_translator.from_relay(relay_mod["main"])
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        relax_mod = relax.transform.FuseOps()(relax_mod)
        relax_mod = relax.transform.FuseTIR()(relax_mod)
        relax_mod = tir.transform.PromoteDataType()(relax_mod)
    return relax_mod


def apply_opt_after_tuning(relax_mod: IRModule, database: ms.database.Database, target: Target):
    with transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyHistoryBest(database, target)(relax_mod)
        if ARGS.target.kind.name != "cuda":
            relax_mod = relax.transform.LayoutRewrite()(relax_mod)
            relax_mod = relax.transform.FoldConstant()(relax_mod)
    return relax_mod


def f_remote_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input_data):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    evaluator = vm.module.time_evaluator(
        func_name="main",
        dev=device,
        repeat=3,
        min_repeat_ms=500,
    )
    # Use millisecond as the unit
    costs = np.mean(np.array(evaluator(*input_data).results)) * 1000.0
    output = vm["main"](*input_data)
    return costs, output.numpy()


def apply_postproc(mod: IRModule, target: Target):
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        postprocs=ms.tune.Parse._postproc(postproc=None, target=target),
        task_name="untuned",
    )
    for rule in ctx.postprocs:
        rule.initialize_with_tune_context(ctx)

    sch = tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    for postproc in ctx.postprocs:
        assert postproc.apply(sch)
    return sch.mod


def run_and_measure(
    mod: tvm.IRModule,
    input_data: np.ndarray,
    params: Dict[str, tvm.runtime.NDArray],
    with_params_bound: bool,
):
    with transform.PassContext(opt_level=3):
        executable = relax.vm.build(mod=mod, target=ARGS.target)

    args = [input_data]
    if not with_params_bound:
        for param in params.values():
            args.append(param.numpy())

    return run_module_via_rpc(
        rpc_config=ARGS.rpc_config,
        lib=executable.mod,
        dev_type=ARGS.target.kind.name,
        args=args,
        continuation=f_remote_measurement,
    )


def main():
    task_name = ARGS.model + "_" + ARGS.layout
    work_dir = ARGS.work_dir

    path_workload = os.path.join(work_dir, f"{task_name}_database_workload.json")
    path_tuning_record = os.path.join(work_dir, f"{task_name}_database_tuning_record.json")
    database = ms.database.JSONDatabase(
        path_workload=path_workload,
        path_tuning_record=path_tuning_record,
    )

    num_layers = 18
    batch_size = 1
    image_shape = (3, 224, 224) if ARGS.layout == "NCHW" else (224, 224, 3)
    input_shape = (batch_size,) + image_shape

    relay_mod, params = tvm.relay.testing.resnet.get_workload(
        num_layers=num_layers,
        image_shape=image_shape,
        batch_size=batch_size,
        layout=ARGS.layout,
        dtype="float32",
    )

    # translate the ResNet model from Relay to Relax
    relax_mod_w_opt = apply_opt_before_tuning(relay_mod=relay_mod, params=params)
    relax_mod_wo_opt = relay_translator.from_relay(relay_mod["main"])
    assert isinstance(relax_mod_w_opt, tvm.IRModule)
    assert isinstance(relax_mod_wo_opt, tvm.IRModule)

    if ARGS.tune_model:
        tune_relax(
            mod=relax_mod_w_opt,
            target=ARGS.target,
            config=EvolutionarySearchConfig(
                num_trials_per_iter=64,
                max_trials_per_task=ARGS.num_trials,
                max_trials_global=ARGS.num_trials,
                init_min_unmeasured=50,
            ),
            runner=ms.runner.RPCRunner(
                rpc_config=ARGS.rpc_config,
                alloc_repeat=3,
                max_workers=ARGS.rpc_workers,
            ),
            database=database,
            task_name=task_name,
            work_dir=work_dir,
            num_threads=os.cpu_count(),
        )

    relax_mod_wo_opt = apply_postproc(relax_mod_wo_opt, ARGS.target)
    relax_mod_w_opt = apply_opt_after_tuning(relax_mod_w_opt, database, ARGS.target)

    input_data = np.random.rand(*input_shape).astype("float32")
    costs_wo_opt, output_wo_opt = run_and_measure(
        relax_mod_wo_opt, input_data, params, with_params_bound=False
    )
    costs_w_opt, output_w_opt = run_and_measure(
        relax_mod_w_opt, input_data, params, with_params_bound=True
    )
    print(f"Without opt: {costs_wo_opt} ms")
    print(f"With opt: {costs_w_opt} ms")
    tvm.testing.assert_allclose(output_wo_opt, output_w_opt, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    main()

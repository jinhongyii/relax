/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/relax/transform/layout_rewrite.cc
 * \brief Perform layout rewrite
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/function.h>

#include "../../printer/text_printer.h"
#include "../../tir/schedule/analysis.h"
#include "tvm/tir/schedule/schedule.h"

namespace tvm {
namespace relax {
using tir::Buffer;
using tir::PrimFunc;

struct LayoutRewriteInfo {
  tir::IndexMap index_map;
  Array<PrimExpr> src_shape;
  Array<PrimExpr> tgt_shape;
  DataType dtype;
};

Array<PrimExpr> NormalizeShapeForRelax(const Array<PrimExpr>& shape) {
  Array<PrimExpr> res;
  for (const auto& e : shape) {
    res.push_back(IntImm(DataType::Int(64), e.as<IntImmNode>()->value));
  }
  return res;
}

class LayoutRewriteInserter : public ExprMutator {
 public:
  LayoutRewriteInserter(IRModule module) : module_(GetRef<IRModule>(module.CopyOnWrite())) {
    InitializeIndexMaps();
  }

  void InitializeIndexMaps() {
    Map<GlobalVar, BaseFunc> funcs = module_->functions;
    for (const std::pair<GlobalVar, BaseFunc>& pr : funcs) {
      if (const auto* f = pr.second.as<tir::PrimFuncNode>()) {
        index_maps[f] = {};

        IRModule tmp_mod({{GlobalVar("main"), GetRef<PrimFunc>(f)}});
        tir::Schedule sch =
            tir::Schedule::Concrete(tmp_mod, -1, 0, tir::ScheduleErrorRenderLevel::kDetail);
        for (const tir::BlockRV& block_rv : sch->GetChildBlocks(sch->GetBlock("root"))) {
          tir::Block block = sch->Get(block_rv);
          if (Optional<ObjectRef> ann = block->annotations.Get("layout_free_placeholders")) {
            auto layout_free_buffers = Downcast<Array<tir::Buffer>>(ann.value());
            sch->Unannotate(block_rv, "layout_free_placeholders");
            Optional<Buffer> buffer;
            int buffer_index = -1;
            int var_index = -1;
            for (int j = 0; j < static_cast<int>(layout_free_buffers.size()); j++) {
              for (int i = 0; i < static_cast<int>(block->reads.size()); i++) {
                if (layout_free_buffers[j].same_as(block->reads[i]->buffer)) {
                  buffer = block->reads[i]->buffer;
                  buffer_index = i;
                  break;
                }
              }
              for (int i = 0; i < static_cast<int>(f->params.size()); i++) {
                if (layout_free_buffers[j].same_as(f->buffer_map.Get(f->params[i]).value())) {
                  var_index = i;
                  break;
                }
              }
              if (buffer.defined()) {
                Array<tir::For> loops;
                for (const tir::LoopRV& loop_rv : sch->GetLoops(block_rv)) {
                  loops.push_back(sch->Get(loop_rv));
                }
                arith::Analyzer analyzer;
                tir::BlockRealize realize =
                    tir::GetBlockRealize(sch->state(), sch->GetSRef(block_rv));
                Optional<tir::IndexMap> index_map =
                    tir::SuggestIndexMap(buffer.value(), realize, loops, &analyzer);
                if (index_map.defined()) {
                  sch->TransformLayout(block_rv, buffer_index, tir::BufferIndexType::kRead,
                                       index_map.value());
                  DataType dtype = buffer.value()->dtype;
                  Array<PrimExpr> src_shape = buffer.value()->shape;
                  Array<PrimExpr> tgt_shape = index_map.value()->MapShape(src_shape);
                  ICHECK(sch->mod()->functions.size() == 1);
                  for (const auto& kv : sch->mod()->functions) {
                    index_maps[(tir::PrimFuncNode*)kv.second.get()].push_back(std::make_pair(
                        var_index,
                        LayoutRewriteInfo{index_map.value(), src_shape, tgt_shape, dtype}));
                    module_->Update(pr.first, kv.second);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  void Mutate() {
    auto funcs = module_->functions;
    for (auto pr : funcs) {
      if (pr.second.as<FunctionNode>()) {
        module_->Update(pr.first, Downcast<BaseFunc>(VisitExpr(pr.second)));
      }
    }
  }

  IRModule GetMod() { return module_; }

 private:
  GlobalVar CreateFuncFromIndexMap(const LayoutRewriteInfo& info) {
    const Buffer& src_buffer = te::decl_buffer(info.src_shape, info.dtype, "src", "global");
    const Buffer& tgt_buffer = te::decl_buffer(info.tgt_shape, info.dtype, "tgt", "global");
    Array<PrimExpr> src_indices;
    for (tir::Var v : info.index_map->initial_indices) {
      src_indices.push_back(v);
    }
    tir::Stmt body = tir::BufferStore(tgt_buffer, tir::BufferLoad(src_buffer, src_indices),
                                      info.index_map->final_indices);
    Array<tir::IterVar> block_iters;
    for (int i = 0; i < static_cast<int>(info.src_shape.size()); i++) {
      block_iters.push_back(tir::IterVar(Range::FromMinExtent(0, info.src_shape[i]),
                                         info.index_map->initial_indices[i], tir::kDataPar));
    }
    Map<String, ObjectRef> annotations;
    annotations.Set(tir::attr::script_parsing_detect_access, IntImm(DataType::Int(32), 3));
    body = tir::Block(block_iters, {}, {}, "layout_rewrite", body, NullOpt, {}, {}, annotations);
    Array<PrimExpr> loop_vars;
    for (int i = 0; i < static_cast<int>(info.src_shape.size()); i++) {
      loop_vars.push_back(info.index_map->initial_indices[i].copy_with_suffix("v"));
    }
    body = tir::BlockRealize(loop_vars, Bool(true), Downcast<tir::Block>(body));
    for (int i = info.src_shape.size() - 1; i >= 0; i--) {
      body = tir::For(Downcast<tir::Var>(loop_vars[i]), 0, info.src_shape[i], tir::ForKind::kSerial,
                      body);
    }
    tir::Var src_arg("src", PrimType(DataType::Handle()));
    tir::Var tgt_arg("tgt", PrimType(DataType::Handle()));
    Array<tir::Var> parameters{src_arg, tgt_arg};
    Map<tir::Var, Buffer> buffer_map{{src_arg, src_buffer}, {tgt_arg, tgt_buffer}};
    String func_name = "layout_rewrite" + std::to_string(cnt++);
    PrimFunc func = WithAttrs(PrimFunc(/*params=*/std::move(parameters),
                                       /*body=*/body,
                                       /*ret_type=*/VoidType(),
                                       /*buffer_map=*/std::move(buffer_map)),
                              {{"global_symbol", func_name}, {"tir.noalias", Bool(true)}});
    const auto* complete = runtime::Registry::Get("script.Complete");
    ICHECK(complete);
    PrimFunc completed_func = (*complete)(func, Array<Buffer>());
    auto global_var = GlobalVar(func_name);
    module_->Add(global_var, completed_func);
    return global_var;
  }

  Expr VisitExpr_(const CallNode* call) override {
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (call->op == call_tir_op) {
      Array<Expr> outs;
      auto func_var = call->args[0];
      PrimFunc func = Downcast<PrimFunc>(module_->functions.Get(Downcast<GlobalVar>(func_var)));
      if (index_maps.count(func.get())) {
        if (call->args[1].as<TupleNode>()) {
          Array<Expr> args = Downcast<Tuple>(call->args[1])->fields;
          for (auto pr : index_maps[func.get()]) {
            GlobalVar layout_rewrite_func = CreateFuncFromIndexMap(pr.second);

            Var new_var = builder_->Emit(
                Call(/*op=*/call_tir_op,
                     /*args=*/
                     {
                         layout_rewrite_func,                                    //
                         Tuple({args[pr.first]}),                                //
                         ShapeExpr(NormalizeShapeForRelax(pr.second.tgt_shape))  //
                     },
                     /*attrs=*/{},
                     /*type_args=*/{DynTensorType(pr.second.tgt_shape.size(), pr.second.dtype)}));
            args.Set(pr.first, new_var);
          }
          return Call(call_tir_op, {call->args[0], Tuple(args), call->args[2]}, {},
                      call->type_args);
        } else {
          Expr arg = call->args[1];
          ICHECK(arg->IsInstance<TupleNode>());
          ICHECK(index_maps[func.get()].size() == 1);
          LayoutRewriteInfo info = index_maps[func.get()][0].second;
          GlobalVar layout_rewrite_func = CreateFuncFromIndexMap(info);

          Var new_var = builder_->Emit(
              Call(call_tir_op,
                   {layout_rewrite_func, arg, ShapeExpr(NormalizeShapeForRelax(info.tgt_shape))},
                   {}, {DynTensorType(info.tgt_shape.size(), info.dtype)}));
          return Call(call_tir_op, {call->args[0], Tuple({new_var}), call->args[2], call->args[3]},
                      {}, call->type_args);
        }
      }
    }
    return GetRef<Expr>(call);
  }

  std::unordered_map<const tir::PrimFuncNode*, std::vector<std::pair<int, LayoutRewriteInfo>>>
      index_maps;
  IRModule module_;
  int cnt = 0;
};

namespace transform {

Pass LayoutRewrite() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    LayoutRewriteInserter inserter(std::move(mod));
    inserter.Mutate();
    return inserter.GetMod();
  };
  return CreateModulePass(pass_func, 0, "VMShapeLower", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LayoutRewrite").set_body_typed(LayoutRewrite);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
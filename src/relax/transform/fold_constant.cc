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

#include <tvm/driver/driver_api.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/relay/interpreter.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

/*!
 * \brief Returns whether \p expr is a literal \p Constant,
 */
bool IsSimpleConstant(const Expr& expr) { return expr->IsInstance<relay::ConstantNode>(); }

class ConstantFolder : public ExprMutator {
  Expr ConstEvaluateTIR(GlobalVar tir_f_var, const Array<Expr>& args, ShapeExpr shape) {
    auto tir_f = module_->functions.Get(tir_f_var).value();
    runtime::Module module;
    if (func_build_cache_.count(tir_f)) {
      module = func_build_cache_.Get(tir_f).value();
    } else {
      module = build(LowerPrimFunc(Downcast<tir::PrimFunc>(tir_f), "tir_function"),
                     eval_cpu_target_, eval_cpu_target_);
      func_build_cache_.Set(tir_f, module);
    }
    TVMRetValue ret;

    //    Array<ObjectRef> args_for_tir = {args.begin(), args.end()};
    std::vector<runtime::NDArray> args_for_tir;
    for (auto arg : args) {
      args_for_tir.push_back(Downcast<relay::Constant>(arg)->data);
    }

    int kArraySize = args_for_tir.size();
    TVMValue values[kArraySize + 1];
    int type_codes[kArraySize + 1];

    DLDevice dev = {DLDeviceType::kDLCPU, 0};
    std::vector<int64_t> shape_values;
    for (const auto v : shape->values) {
      shape_values.push_back(v.as<IntImmNode>()->value);
    }

    runtime::ShapeTuple shape_tuple{shape_values.begin(), shape_values.end()};
    auto ret_tensor = runtime::NDArray::Empty(shape_tuple, DataType::Float(32), dev);
    args_for_tir.push_back(ret_tensor);
    for (int i = 0; i < static_cast<int>(args_for_tir.size()); i++) {
      runtime::TVMArgsSetter(values, type_codes)(i, args_for_tir[i]);
    }
    module.GetFunction("tir_function")
        .CallPacked(TVMArgs(values, type_codes, args_for_tir.size()), &ret);
    return Constant(Downcast<runtime::NDArray>(ObjectRef(ret_tensor)));
  }

 public:
  ConstantFolder(IRModule module) : module_(module) {}

  Expr ForwardBinding(Expr e) {
    if (const auto* v = e.as<VarNode>()) {
      // check that the var is not the input param, otherwise the function will crash
      if (visited_varbinding.count(v)) {
        return LookupBinding(GetRef<Var>(v));
      }
    }
    return e;
  }

  Expr VisitCallTIR(Call call) {
    Array<Expr> args;
    Expr op = call->op;
    if (call->args[1].as<TupleNode>()) {
      args = Downcast<Tuple>(call->args[1])->fields;
    } else {
      args.push_back(call->args[1]);
    }
    op = call->args[0];
    Array<Expr> processed_args;
    for (const auto& e : args) {
      processed_args.push_back(ForwardBinding(e));
    }
    if (!std::all_of(processed_args.begin(), processed_args.end(), IsSimpleConstant)) {
      // At least one non-constant argument.
      return std::move(call);
    }
    // During evaluation we have obviously lost all on_device annotations. However any
    // on_device wrapping this call will be left in place.
    try {
      auto fold_result = ConstEvaluateTIR(Downcast<GlobalVar>(op), processed_args,
                                          Downcast<ShapeExpr>(call->args[2]));
      return fold_result;
    } catch (tvm::Error& error) {
      return std::move(call);
    }
  }

  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Call post_call = Downcast<Call>(VisitExprPostOrder_(call));
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    if (call->op.same_as(call_tir_op)) {
      return VisitCallTIR(post_call);
    } else {
      return std::move(post_call);
    }
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    visited_varbinding.insert(binding->var.get());
    ExprMutator::VisitBinding_(binding);
  }

  Expr VisitExpr_(const DataflowVarNode* op) final {
    Var post_visit = Downcast<Var>(VisitExprPostOrder_(op));
    if (visited_varbinding.count(op)) {
      Expr expr = LookupBinding(GetRef<Var>(op));
      if (expr->IsInstance<relay::ConstantNode>()) {
        return expr;
      }
    }
    return post_visit;
  }
  
  Expr VisitExpr_(const VarNode* op) final {
    Var post_visit = Downcast<Var>(VisitExprPostOrder_(op));
    if (visited_varbinding.count(op)) {
      Expr expr = LookupBinding(GetRef<Var>(op));
      if (expr->IsInstance<relay::ConstantNode>()) {
        return expr;
      }
    }
    return post_visit;
  }

  IRModule module_;
  Target eval_cpu_target_{"llvm"};
  Device eval_cpu_dev_{kDLCPU, /*device_id=*/0};
  std::unordered_set<const VarNode*> visited_varbinding;
  Map<BaseFunc, runtime::Module> func_build_cache_;
};

Expr FoldConstant(const IRModule& m, const Expr& e) {
  Expr expr = ConstantFolder(m).VisitExpr(e);
  return expr;
}

namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FoldConstant(m, f));
      };
  return CreateFunctionPass(pass_func, 0, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FoldConstant").set_body_typed(FoldConstant);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

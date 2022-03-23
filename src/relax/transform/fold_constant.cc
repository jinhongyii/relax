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

#include "../../relay/op/memory/on_device.h"
#include "../../relay/transforms/pattern_utils.h"
namespace tvm {
namespace relax {

/*!
 * \brief Returns whether \p expr is a literal \p Constant, optionally wrapped by an "on_device"
 * annotation CallNode (which serves only to associate an \p VirtualDevice to the constant and has
 * no operational effect).
 */
bool IsSimpleConstant(const Expr& expr) {
  return relay::AsIgnoringOnDevice<ConstantNode>(expr) != nullptr;
}

/*!
 * \brief Returns whether \p expr \p IsSimpleConstant directly or is a tuple of
 * \p IsComplexConstant expressions.
 */
bool IsComplexConstant(const Expr& expr) {
  if (IsSimpleConstant(expr)) {
    return true;
  } else if (const auto* tuple_node = relay::AsIgnoringOnDevice<TupleNode>(expr)) {
    return std::all_of(tuple_node->fields.begin(), tuple_node->fields.end(), IsComplexConstant);
  } else {
    return false;
  }
}

class ConstantFolder : public ExprMutator {
  // Convert value to expression.
  Expr ObjectToExpr(const ObjectRef& value) {
    if (value->IsInstance<runtime::NDArray::ContainerType>()) {
      auto nd_array = Downcast<runtime::NDArray>(value);
      return Constant(nd_array);
    } else if (const auto* val = value.as<runtime::ADTObj>()) {
      runtime::ADT adt = GetRef<runtime::ADT>(val);
      Array<Expr> fields;
      for (size_t i = 0; i < adt.size(); ++i) {
        fields.push_back(ObjectToExpr(adt[i]));
      }
      return Tuple(fields);
    } else {
      LOG(FATAL) << "Cannot handle " << value->GetTypeKey();
      return {};
    }
  }

  Expr ConstEvaluateTIR(GlobalVar tir_f_var, const Array<Expr>& args, ShapeExpr shape) {
    auto tir_f = module_->functions.Get(tir_f_var).value();
    auto module = build(LowerPrimFunc(Downcast<tir::PrimFunc>(tir_f), "tir_function"),
                        eval_cpu_target_, eval_cpu_target_);
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
    return ObjectToExpr(ret_tensor);
  }

 public:
  ConstantFolder(IRModule module) : module_(module) {}

  Expr ForwardBinding(Expr e) {
    if (const auto* v = e.as<VarNode>()) {
      if (memo_.count(GetRef<Var>(v))) {
        return ForwardBinding(memo_.Get(GetRef<Var>(v)).value());
      }
    }
    return e;
  }

  Expr VisitExpr_(const CallNode* call) override {
    // post-order mutation
    Call post_call = Downcast<Call>(VisitExprPostOrder_(call));
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    Array<Expr> args;
    Expr op = call->op;
    if (call->op.same_as(call_tir_op)) {
      if (call->args[1].as<TupleNode>()) {
        args = Downcast<Tuple>(call->args[1])->fields;
      } else {
        args.push_back(call->args[1]);
      }
      op = call->args[0];
    } else {
      return std::move(post_call);
    }
    Array<Expr> processed_args;
    for (const auto& e : args) {
      processed_args.push_back(ForwardBinding(e));
    }
    if (!std::all_of(processed_args.begin(), processed_args.end(), IsComplexConstant)) {
      // At least one non-constant argument.
      return std::move(post_call);
    }
    // During evaluation we have obviously lost all on_device annotations. However any
    // on_device wrapping this call will be left in place.
    auto tmp = ConstEvaluateTIR(Downcast<GlobalVar>(op), processed_args,
                                Downcast<ShapeExpr>(call->args[2]));

    return tmp;
  }

  void VisitBinding_(const VarBindingNode* binding) {
    memo_.Set(binding->var, binding->value);
    ExprMutator::VisitBinding_(binding);
  }

  IRModule module_;
  Target eval_cpu_target_{"llvm"};
  Device eval_cpu_dev_{kDLCPU, /*device_id=*/0};
  Map<Var, Expr> memo_;
};

class Substituter : public ExprMutator {
  void VisitBinding_(const VarBindingNode* binding) {
    memo_.Set(binding->var, binding->value);
    ExprMutator::VisitBinding_(binding);
  }

  Expr VisitExpr_(const VarNode* op) {
    if (auto val = memo_.Get(runtime::GetRef<Var>(op))) {
      if (val.value()->IsInstance<ConstantNode>()) {
        return val.value();
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  Map<Var, Expr> memo_;
};

Expr FoldConstant(const IRModule& m, const Expr& e) {
  Expr expr = ConstantFolder(m).VisitExpr(e);
  expr = Substituter().VisitExpr(expr);
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

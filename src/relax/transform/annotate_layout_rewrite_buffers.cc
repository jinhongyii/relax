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
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt.h>


namespace tvm{
namespace relax{

/*! \brief collect the constants that are accessed only once.*/
class ConstantCollector: public ExprVisitor{
 public:
  static std::unordered_set<const ConstantNode*> GetOneAccessConstants(IRModule mod){
    std::unordered_set<const ConstantNode*> ret;
    ConstantCollector collector;
    for (const auto& pr : mod->functions) {
      if (const auto* relax_f = pr.second.as<FunctionNode>()) {
        collector(GetRef<Function>(relax_f));
      }
      ret.insert(collector.one_access_constants_.begin(), collector.one_access_constants_.end());
    }
    return ret;
  }
 private:
  void VisitExpr_(const ConstantNode* op) final{
    ExprVisitor::VisitExpr_(op);
    if (multi_access_constants_.count(op)) {
      return;
    } else if (one_access_constants_.count(op)) {
      one_access_constants_.erase(op);
      multi_access_constants_.insert(op);
    } else {
      one_access_constants_.insert(op);
    }
  }
  std::unordered_set<const ConstantNode*> one_access_constants_;
  std::unordered_set<const ConstantNode*> multi_access_constants_;
};

class ConstantArgFinder: public ExprVisitor{
 public:
  ConstantArgFinder(const std::unordered_set<const ConstantNode*>& one_access_constants, 
                    const IRModule& mod)
      : one_access_constants_(one_access_constants), mod_(mod) {}

  static IRModule MarkLayoutRewriteAttr(const IRModule& mod){
    auto one_access_constants = ConstantCollector::GetOneAccessConstants(mod);
    ConstantArgFinder finder(one_access_constants, mod);
    for (const auto& pr : mod->functions) {
      if (const auto* relax_f = pr.second.as<FunctionNode>()) {
        finder(GetRef<Function>(relax_f));
      }
    }
    for (const auto& pr : mod->functions) {
      if (const auto* tir_f = pr.second.as<tir::PrimFuncNode>()) {
        auto it = finder.attrs_to_add_.find(GetRef<tir::PrimFunc>(tir_f));
        if (it != finder.attrs_to_add_.end()) {
          mod->Update(pr.first, WithAttr(GetRef<tir::PrimFunc>(tir_f), 
                                         tir::attr::layout_rewrite_buffers, (*it).second));
        }
      }
    }
    return mod;
  }

 private:
  /*!
   * \brief Pattern match op to a TIR function and look it up.
   * \return The TIR function, or nullopt if patter match fails.
   */
  Optional<tir::PrimFunc> MatchPrimFunc(const Expr& op) {
    if (auto* ptr = op.as<GlobalVarNode>()) {
      // NOTE: as check works for nullptr(returns null)
      Optional<BaseFunc> base_func = mod_->functions.Get(GetRef<GlobalVar>(ptr));
      if (auto* pfunc = base_func.as<tir::PrimFuncNode>()) {
        return GetRef<tir::PrimFunc>(pfunc);
      }
    }
    return NullOpt;
  }
  
  void VisitExpr_(const CallNode* op) final{
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    ExprVisitor::VisitExpr_(op);
    if (op->op.same_as(call_tir_op_)) {
      Array<Integer> const_args;
      Array<Expr> call_tir_args = Downcast<Tuple>(op->args[1])->fields;
      for (int i = 0; i < static_cast<int>(call_tir_args.size()); i++) {
        if (const auto* constant = call_tir_args[i].as<ConstantNode>()) {
          if (one_access_constants_.count(constant)) {
            const_args.push_back(i);
          }
        }
      }

      Optional<tir::PrimFunc> opt_f = MatchPrimFunc(op->args[0]);
      if (!opt_f) {
        LOG(FATAL) << "Cannot find the prim_func of the call_tir in the module: "
                   << op->args[0];
      }
      tir::PrimFunc f = opt_f.value();
      auto it = attrs_to_add_.find(f);
      if (it == attrs_to_add_.end()) {
        attrs_to_add_.Set(f, const_args);
      } else {
        Array<Integer> seen_const_args = (*it).second;
        Array<Integer> intersection;
        for (int i = 0; i < static_cast<int>(const_args.size()); i++) {
          for (int j = 0; j < static_cast<int>(seen_const_args.size()); j++) {
            if (const_args[i]->value == seen_const_args[j]->value) {
              intersection.push_back(i);
              break ;
            }
          }
        }
        if (intersection.empty()) {
          attrs_to_add_.erase(f);
        } else {
          attrs_to_add_.Set(f, intersection);
        }
      }
    }
  }
  std::unordered_set<const ConstantNode*> one_access_constants_;
  Map<tir::PrimFunc, Array<Integer>> attrs_to_add_;
  const IRModule& mod_;
};

namespace transform{
  Pass AnnotateLayoutRewriteBuffers() {
    runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
        [=](IRModule m, PassContext pc) {
          return ConstantArgFinder::MarkLayoutRewriteAttr(m);
        };
    return CreateModulePass(/*pass_function=*/pass_func,  //
                            /*opt_level=*/0,              //
                            /*pass_name=*/"AnnotateLayoutRewriteBuffers",      //
                            /*required=*/{});
  }
  
  TVM_REGISTER_GLOBAL("relax.transform.AnnotateLayoutRewriteBuffers").set_body_typed(AnnotateLayoutRewriteBuffers);
} // namespace transform
} //namespace relax
} // namespace tvm
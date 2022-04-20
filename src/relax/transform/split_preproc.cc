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
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>

namespace tvm{

namespace tir {

class PreProcBlockCollector:public StmtExprVisitor{
 public:
  static Map<Integer, PrimFunc> GetPreProcFuncByBufferIndex(const PrimFunc& orig_func){
    PreProcBlockCollector collector(orig_func);
    BlockRealize root_realize = Downcast<BlockRealize>(orig_func->body);
    collector(root_realize->block->body);
    return collector.preproc_funcs_;
  }
 private:
  explicit PreProcBlockCollector(const PrimFunc& orig_func): orig_func_(orig_func){}
  
  void ConstructFunc(const BlockRealizeNode* op){
    ICHECK_EQ(op->block->reads.size(), 1);
    ICHECK_EQ(op->block->writes.size(), 1);
    Stmt body = GetRef<BlockRealize>(op);
    ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*(op->block.get()));
    new_block->annotations.erase("preproc");
    ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*op);
    new_realize->block = Block(new_block);
    body = BlockRealize(new_realize);
    for (int i = static_cast<int>(loops_.size()) - 1; i >= 0; i--) {
      ObjectPtr<ForNode> new_for = make_object<ForNode>(*(loops_[i].get()));
      new_for->body = body;
      body = For(new_for);
    }
    body = Block({},{},{},"root",body);
    body = BlockRealize({}, Bool(true), Downcast<Block>(body));
    tir::Var src_arg("src", PrimType(DataType::Handle()));
    tir::Var tgt_arg("tgt", PrimType(DataType::Handle()));
    Array<tir::Var> parameters{src_arg, tgt_arg};
    Buffer src_buffer = op->block->reads[0]->buffer;
    Buffer tgt_buffer = op->block->writes[0]->buffer;
    Map<tir::Var, Buffer> buffer_map{{src_arg, src_buffer}, {tgt_arg, tgt_buffer}};
    PrimFunc func =PrimFunc(/*params=*/std::move(parameters),
                             /*body=*/body,
                             /*ret_type=*/VoidType(),
                             /*buffer_map=*/std::move(buffer_map));
    for (int i = 0; i < static_cast<int>(orig_func_->params.size()); i++) {
      if (src_buffer.same_as(orig_func_->buffer_map.Get(orig_func_->params[i]).value())) {
        ICHECK(!preproc_funcs_.count(i));
        preproc_funcs_.Set(i, func);
        return ;
      }
    }
    LOG(FATAL)<<"ValueError: The read buffer of preproc block is not in the primfunc's buffer map.";
  }
  
  void VisitStmt_(const ForNode* op) final{
    loops_.push_back(GetRef<For>(op));
    StmtVisitor::VisitStmt_(op);
    loops_.pop_back();
  }
  
  void VisitStmt_(const BlockRealizeNode* op) final{
    //preproc block will only appear under root block, so we don't recursively visit
    Block block = op->block;
    auto it = block->annotations.find("preproc");
    if (it != block->annotations.end()  && is_one(Downcast<PrimExpr>((*it).second))) {
      ConstructFunc(op);
    }
  }
  
  const PrimFunc& orig_func_;
  Array<For> loops_;
  Map<Integer, PrimFunc> preproc_funcs_;
};
} // namespace tir

namespace relax{

class SplitPreProcMutator: public ExprMutator{
 public:
  static IRModule SplitPreProc(IRModule mod){
    SplitPreProcMutator mutator(mod);
    for (const auto& pr : mod->functions) {
      if (pr.second.as<FunctionNode>()) {
        auto func = Downcast<Function>(mutator(pr.second));
        mutator.builder_->AddFuncToContext(func, pr
                                                                                       .first->name_hint);
      } else {
        mutator.builder_->GetContextIRModule()->Update(pr.first, pr.second);
      }
    }
    mod =  mutator.builder_->GetContextIRModule();
    mod = tir::transform::RemoveWeightLayoutRewriteBlock()(mod);
    return mod;
  }
  
 private:
  explicit SplitPreProcMutator(const IRModule& mod):mod_(mod){}
  
  /*!
   * \brief Pattern match op to a TIR function and look it up.
   * \return The TIR function, or nullopt if pattern match fails.
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
  
  Array<Expr> GetTIRArgs(const CallNode* call){
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    ICHECK_EQ(call->op, call_tir_op);
    Array<Expr> ret;
    if (call->args[1].as<TupleNode>()) {
      ret = Downcast<Tuple>(call->args[1])->fields;
    } else {
      ret = {call->args[1]};
    }
    return ret;
  }

  ShapeExpr GetOutputShapeFromPreProcFunc(const tir::PrimFunc& func){
    ICHECK_EQ(func->params.size(),2);
    tir::Buffer buffer = func->buffer_map.Get(func->params[1]).value();
    return ShapeExpr(buffer->shape);
  }

  Expr VisitExpr_(const CallNode* call) final{
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (call->op == call_tir_op) {
      Optional<tir::PrimFunc> func = MatchPrimFunc(call->args[0]);
      if (func) {
        Map<Integer, tir::PrimFunc> preproc_func_map =
            tir::PreProcBlockCollector::GetPreProcFuncByBufferIndex(func.value());
        Array<Expr> tir_args = GetTIRArgs(call);
        for (const auto& pr : preproc_func_map) {
          int idx = pr.first;
          ICHECK_LE(idx, tir_args.size());
          tir::PrimFunc func = pr.second;
          ICHECK_EQ(func->params.size(),2);
          tir::Buffer buffer = func->buffer_map.Get(func->params[1]).value();
          ShapeExpr out_shape = GetOutputShapeFromPreProcFunc(func);
          GlobalVar layout_rewrite_func = builder_->AddFuncToContext(pr.second, "layout_rewrite_"+
                                                                        std::to_string(cnt_++));
          Var new_var = builder_->Emit(
              Call(/*op=*/call_tir_op,
                   /*args=*/
                   {
                       layout_rewrite_func,                                    //
                       Tuple({tir_args[idx]}),                                //
                       out_shape  //
                   },
                   /*attrs=*/{},
                   /*type_args=*/{DynTensorType(buffer->shape.size(), buffer->dtype)}));
          tir_args.Set(idx, new_var);
        }
        return Call(call_tir_op, {call->args[0], Tuple(tir_args), call->args[2]}, {},
                    call->type_args);
      }
    }
    return GetRef<Expr>(call);
  }
  
  const IRModule& mod_;
  int cnt_ = 0;
  
};
namespace transform{
Pass SplitPreProc() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) {
        return SplitPreProcMutator::SplitPreProc(m);
      };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"SplitPreProc",      //
                          /*required=*/{});
}
  
TVM_REGISTER_GLOBAL("relax.transform.SplitPreProc").set_body_typed(SplitPreProc);
} // namespace transform
} //namespace relax
} // namespace tvm
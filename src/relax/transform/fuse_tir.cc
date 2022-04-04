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
#include <tvm/tir/stmt_functor.h>

#include "../../relay/analysis/graph_partitioner.h"
#include "../../support/arena.h"
#include "../../tir/ir/functor_common.h"

namespace tvm {
namespace tir {

/*!
 * \brief Substitute a given source buffer with a given target buffer in statements or expressions
 */
class BufferSubstituter : private StmtExprMutator {
 public:
  static Stmt Substitute(const Map<Buffer, Buffer>& buffer_map, Stmt stmt) {
    return BufferSubstituter(buffer_map)(std::move(stmt));
  }

 private:
  explicit BufferSubstituter(const Map<Buffer, Buffer>& buffer_map) {
    for (const auto& kv : buffer_map) {
      const Buffer& src = kv.first;
      const Buffer& tgt = kv.second;
      buffer_var_map_[src->data.get()] = tgt;
    }
  }

  PrimExpr VisitExpr_(const VarNode* _op) final {
    auto it = buffer_var_map_.find(_op);
    if (it != buffer_var_map_.end()) {
      return it->second->data;
    } else {
      return GetRef<PrimExpr>(_op);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    auto it = buffer_var_map_.find(load->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      auto n = make_object<BufferLoadNode>(*load.get());
      n->buffer = it->second;
      return BufferLoad(n);
    } else {
      return std::move(load);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    auto it = buffer_var_map_.find(store->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      auto n = CopyOnWrite(store.get());
      n->buffer = it->second;
      return BufferStore(n);
    } else {
      return std::move(store);
    }
  }

  PrimExpr VisitExpr_(const LoadNode* _op) final {
    Load load = Downcast<Load>(StmtExprMutator::VisitExpr_(_op));
    auto it = buffer_var_map_.find(load->buffer_var.get());
    if (it != buffer_var_map_.end()) {
      auto n = make_object<LoadNode>(*load.get());
      n->buffer_var = it->second->data;
      return Load(n);
    } else {
      return std::move(load);
    }
  }

  Stmt VisitStmt_(const StoreNode* _op) final {
    Store store = Downcast<Store>(StmtExprMutator::VisitStmt_(_op));
    auto it = buffer_var_map_.find(store->buffer_var.get());
    if (it != buffer_var_map_.end()) {
      auto n = CopyOnWrite(store.get());
      n->buffer_var = it->second->data;
      return Store(n);
    } else {
      return std::move(store);
    }
  }

  Stmt VisitStmt_(const BlockNode* _op) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(_op));

    // Define the mutation functions.
    auto f_mutate_match_buffers = [this](const MatchBufferRegion& match_buffer) {
      const Buffer& src_buffer = match_buffer->source->buffer;
      auto it = buffer_var_map_.find(src_buffer->data.get());
      if (it != buffer_var_map_.end()) {
        return MatchBufferRegion(match_buffer->buffer,
                                 BufferRegion(it->second, match_buffer->source->region));
      } else {
        return match_buffer;
      }
    };

    auto f_mutate_read_write_region = [this](const BufferRegion& buffer_region) {
      auto it = buffer_var_map_.find(buffer_region->buffer->data.get());
      return it == buffer_var_map_.end() ? buffer_region
                                         : BufferRegion(it->second, buffer_region->region);
    };

    // Step 1. Mutate `match_buffers`.
    Array<MatchBufferRegion> match_buffers =
        MutateArray(block->match_buffers, f_mutate_match_buffers);
    // Step 2. Mutate the read/write region.
    Array<BufferRegion> reads = MutateArray(block->reads, f_mutate_read_write_region);
    Array<BufferRegion> writes = MutateArray(block->writes, f_mutate_read_write_region);

    reads = UnionAccessRegion(reads);
    writes = UnionAccessRegion(writes);

    if (reads.same_as(block->reads) &&    //
        writes.same_as(block->writes) &&  //
        match_buffers.same_as(block->match_buffers)) {
      return std::move(block);
    } else {
      auto n = CopyOnWrite(block.get());
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->match_buffers = std::move(match_buffers);
      return Block(n);
    }
  }

 private:
  /*! \brief Mapping from src buffer.data to tgt buffer. */
  std::unordered_map<const tir::VarNode*, tir::Buffer> buffer_var_map_;

  static Array<tir::BufferRegion> UnionAccessRegion(Array<BufferRegion> regions) {
    // For now we only allow Buffer access the same elements.
    // e.g. `[A[vi, vj], A[vi, vj]]` is a legal pattern but need to union to `A[vi, vj]`
    // However, `A[vi, vj], A[vi, vj + 1]` is not allow for now.
    Array<BufferRegion> ret;
    std::unordered_map<const BufferNode*, Region> buffer_region_set;

    for (const BufferRegion& region : regions) {
      auto it = buffer_region_set.find(region->buffer.get());
      if (it == buffer_region_set.end()) {
        ret.push_back(region);
        buffer_region_set[region->buffer.get()] = region->region;
      } else {
        ICHECK(tvm::StructuralEqual()(region->region, it->second));
      }
    }

    if (ret.size() == regions.size()) {
      return regions;
    } else {
      return ret;
    }
  }
};

/*!
 * \brief A mutator which detect block name duplication and deduplicate the names.
 */
class BlockNameDeduplicator : public tir::StmtMutator {
 private:
  Stmt VisitStmt_(const BlockNode* block) final {
    // Step 1. Get the unique name of this block.
    String name = GetUniqueName(block->name_hint);

    // Step 2. Recursively mutate the init and the body of this block.
    Optional<Stmt> init = NullOpt;
    if (block->init.defined()) {
      init = VisitStmt(block->init.value());
    }
    Stmt body = VisitStmt(block->body);

    // Step 3. If everything remains unchanged, return the original block. Otherwise return the new
    // block.
    if (name == block->name_hint && init.same_as(block->init) && body.same_as(block->body)) {
      return GetRef<Block>(block);
    } else {
      ObjectPtr<BlockNode> n = CopyOnWrite(block);
      n->name_hint = std::move(name);
      n->body = std::move(body);
      n->init = std::move(init);
      return Stmt(n);
    }
  }

  String GetUniqueName(const String& prefix) {
    String unique_prefix = prefix;
    auto it = name_count_.find(prefix);
    while (name_count_.count(unique_prefix)) {
      unique_prefix = prefix + "_" + std::to_string(++it->second);
    }
    name_count_[unique_prefix] = 0;
    return unique_prefix;
  }

  /*! \brief The count map to make block name unique. */
  std::unordered_map<String, int> name_count_;
};

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace relax {

/*! \brief auxiliary information for FuseTIR */
struct FuseFuncInfo {
  FuseFuncInfo() = default;

  /*! \brief The arguments for calling prim_func */
  Array<Expr> arguments;
  /*!
   * \brief The map from each dataflow var (intermediate var) to the corresponding buffers
   * allocated in the fused func
   */
  Map<Var, Array<tir::Buffer>> var2buffers;
  /*! \brief The buffers to allocate in the fused func*/
  Array<tir::Buffer> alloc_buffers;
  /*!
   * \brief The bodies of the original funcs, which is also the body of the fused func after
   * flattening
   */
  Array<tir::Stmt> bodies;
  /*! \brief The params of the fused function*/
  Array<tir::Var> params;
  /*!
   * \brief The map from buffer in original functions to corresponding buffer in the fused
   * function
   */
  Map<tir::Buffer, tir::Buffer> buffer_subst_map;
  /*! \brief The `buffer_map` in the fused function*/
  Map<tir::Var, tir::Buffer> buffer_map;
  /*! \brief The name of the fused function */
  std::string global_name = "fused";
};
/*!
 * \brief Construct a fused TIR function
 */
class FusedTIRConstructor : public ExprVisitor {
 public:
  /*!
   * \brief Construct a fused TIR function from a subfunction
   * \param mod The IRModule
   * \param func The subfunction
   * \return The fused TIR function
   */
  static tir::PrimFunc GetFusedTIR(const IRModule& mod, const BaseFunc& func) {
    FusedTIRConstructor visitor(mod);
    visitor(func);
    return visitor.fused_tir_;
  }

 private:
  explicit FusedTIRConstructor(const IRModule& mod) : mod_(mod) {}

  /*! \brief auxiliary information for FuseTIR */
  struct FuseFuncInfo {
    FuseFuncInfo() = default;

    /*! \brief The arguments for calling prim_func */
    Array<Expr> arguments;
    /*!
     * \brief The map from each dataflow var (intermediate var) to the corresponding buffers
     * allocated in the fused func
     */
    Map<Var, Array<tir::Buffer>> var2buffers;
    /*! \brief The buffers to allocate in the fused func*/
    Array<tir::Buffer> alloc_buffers;
    /*!
     * \brief The bodies of the original funcs, which is also the body of the fused func after
     * flattening
     */
    Array<tir::Stmt> bodies;
    /*! \brief The params of the fused function*/
    Array<tir::Var> params;
    /*!
     * \brief The map from buffer in original functions to corresponding buffer in the fused
     * function
     */
    Map<tir::Buffer, tir::Buffer> buffer_subst_map;
    /*! \brief The `buffer_map` in the fused function*/
    Map<tir::Var, tir::Buffer> buffer_map;
    /*! \brief The name of the fused function */
    std::string global_name = "fused";
  };

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

  /*!
   * \brief Map the old output buffer to the new one in the buffer map, and then update
   * buffer_subst_map
   * \param call The call_tir
   * \param func The TIR func
   */
  void MapOutputBufferAndUpdateBufferSubstMap(const CallNode* call, const tir::PrimFunc& func) {
    const Expr& output_shapes = call->args[2];
    int output_size;
    if (const auto* tuple_output_shapes = output_shapes.as<TupleNode>()) {
      output_size = tuple_output_shapes->fields.size();
    } else {
      output_size = 1;
    }
    int n = func_info_.params.size();
    for (int i = 0; i < output_size; i++) {
      tir::Buffer buffer =
          func_info_.buffer_map.Get(func_info_.params[n - output_size + i]).value();
      tir::Buffer old_buffer =
          func->buffer_map.at(func->params[func->params.size() - output_size + i]);
      func_info_.buffer_subst_map.Set(old_buffer, buffer);
    }
  }

  /*!
   * \brief Allocate buffer(s) for the output var and then update var2buffers
   * \param call The call_tir
   * \param func The TIR func
   * \param output_var The output var
   */
  void AllocateOutputBufferAndUpdateVar2Buffers(const CallNode* call, const tir::PrimFunc& func,
                                                Var output_var) {
    const Expr& output_shapes = call->args[2];
    int output_size;
    if (const auto* tuple_output_shapes = output_shapes.as<TupleNode>()) {
      output_size = tuple_output_shapes->fields.size();
    } else {
      output_size = 1;
    }
    int n = func->params.size();
    // Allocate output buffer
    Array<tir::Buffer> alloc_buffers;
    for (int i = 0; i < output_size; i++) {
      tir::Buffer buffer = func->buffer_map.at(func->params[i + n - output_size]);
      func_info_.alloc_buffers.push_back(buffer);
      alloc_buffers.push_back(buffer);
    }
    // Update var2buffers
    func_info_.var2buffers.Set(output_var, alloc_buffers);
  }

  /*!
   * \brief Create an array buffers with specified type and shape
   * \param The specified type, can be DynTensorType or TupleType
   * \param The specified shape, can be ShapeExpr or Tuple
   * \return The created array of buffers
   */
  Array<tir::Buffer> CreateBuffers(Type type, Expr shape) {
    Array<tir::Buffer> ret;
    if (const relax::DynTensorTypeNode* tty = type.as<relax::DynTensorTypeNode>()) {
      const auto* shape_expr = shape.as<ShapeExprNode>();
      ICHECK(shape_expr);
      tir::Buffer buffer = tir::decl_buffer(shape_expr->values, tty->dtype);
      ret.push_back(buffer);
    } else if (const TupleTypeNode* tty = type.as<TupleTypeNode>()) {
      const auto* tuple_shape = shape.as<TupleNode>();
      ICHECK(tuple_shape);
      for (int i = 0; i < static_cast<int>(tty->fields.size()); i++) {
        const auto* dyn_ty = tty->fields[i].as<relax::DynTensorTypeNode>();
        ICHECK(tty);
        const auto* shape_expr = tuple_shape->fields[i].as<ShapeExprNode>();
        ICHECK(shape_expr);
        tir::Buffer buffer = tir::decl_buffer(shape_expr->values, dyn_ty->dtype);
        ret.push_back(buffer);
      }
    } else {
      LOG(FATAL) << "ValueError: wrong param type. Got: " << type;
    }
    return ret;
  }

  /*!
   * \brief Map old TIR func param buffer to new buffer, and then update `buffer_subst_map`
   * \param call The call_tir
   * \param func The TIR func
   */
  void MapArgsToBufferAndUpdateBufferSubstMap(const CallNode* call, const tir::PrimFunc& func) {
    Array<Expr> arg_fields;
    if (const auto* tuple_args = call->args[1].as<TupleNode>()) {
      arg_fields = tuple_args->fields;
    } else {
      arg_fields = {call->args[1]};
    }
    for (int i = 0; i < static_cast<int>(arg_fields.size()); i++) {
      tir::Buffer buffer = func->buffer_map.at(func->params[i]);
      // Check the buffer has const shape
      for (PrimExpr e : buffer->shape) {
        ICHECK(e.as<IntImmNode>());
      }
      if (const auto* v = arg_fields[i].as<VarNode>()) {
        auto it = func_info_.var2buffers.find(GetRef<Var>(v));
        // Substitute the buffer with the already allocated one if it is an intermediate var
        if (it != func_info_.var2buffers.end()) {
          ICHECK((*it).second.size() == 1);
          func_info_.buffer_subst_map.Set(buffer, (*it).second[0]);
          continue;
        }
      }
    }
  }

  /*!
   * \brief Construct fused TIR func with collected FuseFuncInfo
   * \return The fused TIR
   */
  tir::PrimFunc ConstructFunc() {
    Map<String, ObjectRef> attr_map;
    attr_map.Set("tir.noalias", tir::const_true());
    if (func_info_.global_name != "fused") {
      attr_map.Set("global_symbol", String(func_info_.global_name));
    }
    tir::Stmt body =
        tir::BlockNameDeduplicator().operator()(tir::SeqStmt::Flatten(func_info_.bodies));
    body = tir::BufferSubstituter::Substitute(func_info_.buffer_subst_map, body);
    body = tir::Block({}, {}, {}, "root", std::move(body), NullOpt, func_info_.alloc_buffers);
    body = tir::BlockRealize({}, Bool(true), Downcast<tir::Block>(body));
    tir::PrimFunc func(func_info_.params, body, VoidType(), func_info_.buffer_map,
                       Optional<Map<tir::Var, tir::Buffer>>(), DictAttrs(attr_map));
    return func;
  }

  void VisitExpr_(const FunctionNode* op) final {
    // creating buffers for func params
    for (const Var& param : op->params) {
      Type type = param->checked_type_;
      if (!type.defined()) {
        type = param->type_annotation.value_or(Type());
      }
      if (type.defined()) {
        Array<tir::Buffer> buffers = CreateBuffers(type, param->shape());
        for (int i = 0; i < static_cast<int>(buffers.size()); i++) {
          tir::Var var;
          if (buffers.size() == 1) {
            var = tir::Var(param->name_hint(), PrimType(DataType::Handle()));
          } else {
            var = tir::Var(param->name_hint() + "_" + std::to_string(i),
                           PrimType(DataType::Handle()));
          }
          func_info_.buffer_map.Set(var, buffers[i]);
          func_info_.params.push_back(var);
        }
        func_info_.var2buffers.Set(param, buffers);
      } else {
        LOG(FATAL) << "The param " << param->name_hint() << " has no type";
      }
    }
    // creating buffer for func output
    Type ret_type = op->ret_type;
    Expr shape;
    if (const relax::SeqExprNode* body = op->body.as<relax::SeqExprNode>()) {
      shape = body->body->shape();
    } else {
      shape = op->body->shape();
    }
    ICHECK(op->name);
    Array<tir::Buffer> buffers = CreateBuffers(ret_type, shape);
    String name_prefix = op->name.value()->name_hint + "_output";
    for (int i = 0; i < static_cast<int>(buffers.size()); i++) {
      tir::Var var;
      if (buffers.size() == 1) {
        var = tir::Var(name_prefix, PrimType(DataType::Handle()));
      } else {
        var = tir::Var(name_prefix + "_" + std::to_string(i), PrimType(DataType::Handle()));
      }
      func_info_.buffer_map.Set(var, buffers[i]);
      func_info_.params.push_back(var);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    this->VisitExpr(binding->value);
    if (const auto* call = binding->value.as<CallNode>()) {
      bool is_dataflow_var = binding->var->IsInstance<DataflowVarNode>();
      if (call->op.same_as(call_tir_op_)) {
        Optional<tir::PrimFunc> opt_f = MatchPrimFunc(call->args[0]);
        if (!opt_f) {
          LOG(FATAL) << "Cannot find the prim_func of the call_tir in the module: "
                     << call->args[0];
        }
        tir::PrimFunc func = opt_f.value();
        // Regenerate all vars/buffer and blocks to avoid duplication
        func = tir::RenewDefs(func);
        // Check functions are all schedulable funcs. i.e. the body of func is root block
        CHECK(func->body->IsInstance<tir::BlockRealizeNode>())
            << "Only schedulable functions (whose body is the root block) can be fused";
        tir::BlockRealize root_realize = Downcast<tir::BlockRealize>(func->body);
        tir::Block root_block = root_realize->block;
        // Add all the original alloc_buffers and body
        func_info_.alloc_buffers.insert(func_info_.alloc_buffers.end(),
                                        root_block->alloc_buffers.begin(),
                                        root_block->alloc_buffers.end());
        func_info_.bodies.push_back(root_block->body);

        // Map input arguments to buffer
        MapArgsToBufferAndUpdateBufferSubstMap(call, func);
        if (is_dataflow_var) {
          // Allocate buffer if it is an intermediate var in the fused function,
          AllocateOutputBufferAndUpdateVar2Buffers(call, func, binding->var);
        } else {
          // Map old output buffer to the new one in the buffer map
          MapOutputBufferAndUpdateBufferSubstMap(call, func);
        }
        // Update fused func name
        Optional<String> name = func->attrs.GetAttr<String>("global_symbol");
        if (name.defined()) func_info_.global_name += "_" + name.value();
      }
      if (!is_dataflow_var) {
        // Construct the fused func
        fused_tir_ = ConstructFunc();
      }
    } else if (const auto* tuple_get_item = binding->value.as<TupleGetItemNode>()) {
      ICHECK(binding->var->IsInstance<DataflowVarNode>())
          << "Currently TupleGetItem outputs are not allowed";
      const Var& tuple_var = Downcast<Var>(tuple_get_item->tuple);
      auto it = func_info_.var2buffers.find(tuple_var);
      if (it != func_info_.var2buffers.end()) {
        func_info_.var2buffers.Set(binding->var, {(*it).second[tuple_get_item->index]});
      }
    } else {
      LOG(FATAL) << "Unsupported binding value: " << binding->value;
    }
  }

  /*! \brief The IRModule */
  const IRModule& mod_;
  /*! \brief The helper info to fuse TIR primfunc */
  FuseFuncInfo func_info_;
  /*! \brief The tir function after fusion*/
  tir::PrimFunc fused_tir_;
};

/*!
 * \brief The helper class to fuse TIR functions and build a new module which calls the fused TIR.
 */
class TIRFuseMutator : public ExprMutator {
 public:
  static IRModule Transform(const IRModule& mod) {
    TIRFuseMutator mutator(mod);
    BaseFunc main_func = mod->Lookup("main");
    mutator.builder_->AddFuncToContext(Downcast<BaseFunc>(mutator.VisitExpr(main_func)), "main");
    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit TIRFuseMutator(const IRModule& mod) : mod_(mod) {}

  Expr VisitExpr_(const CallNode* call) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    Expr e = ExprMutator::VisitExpr_(call);
    call = e.as<CallNode>();
    ICHECK(call != nullptr);
    if (call->op->IsInstance<GlobalVarNode>()) {
      // Handle subfunction
      GlobalVar gv = Downcast<GlobalVar>(call->op);
      BaseFunc func = mod_->Lookup(gv);
      tir::PrimFunc fused_tir = FusedTIRConstructor::GetFusedTIR(mod_, func);
      String name = fused_tir->attrs.GetAttr<String>("global_symbol").value_or("fused");
      GlobalVar fused_tir_gv = this->builder_->AddFuncToContext(fused_tir, name);
      Array<Expr> arg_list;
      for (const Expr& arg : call->args) {
        // flatten input arg list
        if (const auto* tuple_shape = arg->shape().as<TupleNode>()) {
          for (int i = 0; i < static_cast<int>(tuple_shape->fields.size()); i++) {
            Expr new_arg = builder_->Emit(TupleGetItem(arg, i));
            arg_list.push_back(new_arg);
          }
        } else {
          arg_list.push_back(arg);
        }
      }
      Array<Expr> call_args = {fused_tir_gv, Tuple(arg_list), call->shape()};
      Array<Type> call_type_args;
      if (const auto* type = call->checked_type().as<TupleTypeNode>()) {
        call_type_args = type->fields;
      } else {
        call_type_args = {call->checked_type()};
      }
      Call new_call_tir(call_tir_op_, call_args, call->attrs, call_type_args);
      return new_call_tir;
    }

    if (call->op != call_tir_op_) {
      return e;
    }
    // Handle call_tir in main function
    GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
    tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
    GlobalVar new_gv = this->builder_->AddFuncToContext(func, gv->name_hint);
    return Call(call->op, {new_gv, call->args[1], call->args[2]}, call->attrs, call->type_args,
                call->span);
  }

 private:
  /*! \brief The IRModule */
  const IRModule& mod_;
};

IRModule FuseTIR(IRModule mod) {
  mod = TIRFuseMutator::Transform(mod);
  return mod;
}

namespace transform {

Pass FuseTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return relax::FuseTIR(m); };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"FuseTIR",      //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.FuseTIR").set_body_typed(FuseTIR);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

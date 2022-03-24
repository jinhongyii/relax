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
 * \file src/relax/transform/fma_rewrite.cc
 * \brief Perform fused multiply add rewriting in dataflow blocks.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../../relay/analysis/graph_partitioner.h"
#include "../../support/arena.h"
namespace tvm {
namespace relax {

/*
  Note on Fusing algorithm:

  The main challenge of general fusor is to handle possible diamond shape branches,
  in the following graph, conv2d can be fused to elemwise add.

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |

  However, at the point of conv2d we do not necessarily know that all the future paths
  will merge at the elemwise add. The fusion algorithm applies post-dominator analysis.

  The immediate post-dominator of a node defined by the closest node where all the future path goes
  into. In the above case, the elemwise add is the post-dominator of conv2d. The general algorithm
  is as follows:

  - Construct a DAG of dataflow graph for dominator analysis
  - Construct a post-dominator tree which gives immediate post dominator of each node.
  - Run fusion algorithm with the given post-dominator information.

  Note that, because we run analysis on a DAG, we use a single pass post-dominator
  tree construction algorithm via LCA, which is simpler than the full version that handles cycles.

  The fusion algorithm traverses from each node and checks if it can be fused to its
  immediate post dominator. It has to check the following things:

  - CheckPath: check all the path between a node and its immediate post-dominator
               satisfies the fuse condition.
  - Note that these intermediate node can already be fused with another nodes, the algorithm
      will still run correctly.
  - CommitFuse: mark all the nodes between source and post-dominator as the same group.
  - We use an Union-Find data structure to manage the groups.
*/

using relay::GraphPartitioner;
using relay::IndexedForwardGraph;
using relay::OpPatternKind;
using support::LinkedList;
using support::LinkNode;

constexpr uint32_t kMaxFusedOps = 256;

TVM_REGISTER_PASS_CONFIG_OPTION("relax.FuseOps.max_depth", Integer);

// Creator of post dominator tree of the dataflow
class IndexedForwardGraphCreator : private ExprVisitor {
 public:
  static IndexedForwardGraph Create(const IRModule& mod, support::Arena* arena) {
    GlobalVar main_global_var = mod->GetGlobalVar("main");
    Function body = Downcast<Function>(mod->Lookup(main_global_var));
    IndexedForwardGraphCreator creator(arena, mod);
    for (const auto& kv : mod->functions) {
      const Expr& func = kv.second;
      if (func->IsInstance<FunctionNode>()) {
        creator.VisitExpr(body);
      }
    }
    // creator.graph_.DebugDump();
    return creator.graph_;
  }

 private:
  void VisitExpr_(const ConstantNode* op) final {
    this->CreateNode(op);
    this->AddNode(op);
    IndexedForwardGraph::Node* node = graph_.node_map.at(op);
    DataType dtype = DataType(op->data->dtype);
    // This rule must be consistent with code generator.
    bool is_simple_const =
        (dtype == DataType::Int(32) || dtype == DataType::Int(64) || dtype == DataType::Float(32) ||
         dtype == DataType::Float(64) || dtype == DataType::Bool());
    if (op->is_scalar() && is_simple_const) {
      node->pattern = OpPatternKind::kElemWise;
    } else {
      // for now, mark non-scalar constant
      // as opaque, we will not choose to fuse it.
      node->pattern = OpPatternKind::kOpaque;
    }
  }

  void VisitExpr_(const FunctionNode* op) final {
    for (const Var& param : op->params) {
      CreateNode(param.get());
      this->UpdateEdge(param, nullptr, OpPatternKind::kOpaque);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode* op) final { this->AddNode(op); }

  void VisitBindingBlock_(const BindingBlockNode* block) final {
    // Skip Binding Block since it's imprue (with side effect or control flow)
    return;
  }

  void VisitBinding_(const MatchShapeNode* binding) final {
    auto node = CreateNode(binding->var.get());
    this->UpdateEdge(binding->var, node, OpPatternKind::kInjective);
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    // Don't allow recursive var binding.
    ICHECK(!cur_binding_var_.defined());
    cur_binding_var_ = binding->var;
    CreateNode(binding->var.get());
    if (!binding->var->IsInstance<DataflowVarNode>()) {
      this->UpdateEdge(binding->var, nullptr, OpPatternKind::kOpaque);
    }
    ExprVisitor::VisitBinding_(binding);
    cur_binding_var_ = NullOpt;
  }

  void VisitExpr_(const CallNode* op) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");

    ICHECK(cur_binding_var_.defined());
    const Var& binding_var = cur_binding_var_.value();
    auto it = graph_.node_map.find(binding_var.get());
    ICHECK(it != graph_.node_map.end());
    IndexedForwardGraph::Node* node = it->second;

    // If the pattern is not annotated we will default to opaque.
    OpPatternKind op_pattern = OpPatternKind::kOpaque;
    ICHECK(op->op->IsInstance<OpNode>());
    if (op->op == call_tir_op_) {
      GlobalVar global_var = Downcast<GlobalVar>(op->args[0]);
      tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(global_var));
      const Tuple& args = Downcast<Tuple>(op->args[1]);
      const Expr& shape = op->args[2];
      int func_pattern = func->GetAttr<Integer>("op_pattern").value_or(OpPatternKind::kOpaque);

      // TODO(siyuan): Check the integer data is valid
      op_pattern = static_cast<OpPatternKind>(func_pattern);

      for (const Expr& arg : args->fields) {
        this->VisitExpr(arg);
        this->UpdateEdge(arg, node, op_pattern);
      }
    } else {
      LOG(FATAL) << "The call op " << op->op << " is not supported in dataflow block for now.";
    }
    node->pattern = op_pattern;
    this->AddNode(binding_var.get());
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    ICHECK(cur_binding_var_.defined());
    const Var& binding_var = cur_binding_var_.value();
    auto it = graph_.node_map.find(binding_var.get());
    ICHECK(it != graph_.node_map.end());
    IndexedForwardGraph::Node* node = it->second;

    node->pattern = OpPatternKind::kInjective;
    this->UpdateEdge(op->tuple, node, OpPatternKind::kInjective);
    this->AddNode(binding_var.get());
  }

  void VisitExpr_(const IfNode* op) final {
    LOG(FATAL) << "Dataflow block expects no Control flow inside.";
  }

 private:
  explicit IndexedForwardGraphCreator(support::Arena* arena, const IRModule& mod)
      : mod_(mod), arena_(arena) {}

  // Helper functions to maintain IndexedForwardGraph
  /*!
   * \brief The
   * \param node The Relax IR nodes
   * \param parent The parent node in the IndexedForwardGraph.
   *               The source is external if the parent is nullptr.
   * \param pattern The relation pattern between the node and its parent.
   */
  void UpdateEdge(const Expr& node, IndexedForwardGraph::Node* parent, OpPatternKind pattern) {
    const tvm::Object* key = node.get();
    auto it = graph_.node_map.find(key);
    ICHECK(it != graph_.node_map.end());
    IndexedForwardGraph::Node* current = it->second;
    if (parent != nullptr) {
      auto* link = arena_->make<LinkNode<IndexedForwardGraph::Edge>>();
      link->value.node = parent;
      link->value.pattern = pattern;
      current->outputs.Push(link);
    } else {
      current->extern_ref = true;
    }
  }

  void AddNode(const tvm::Object* key) {
    auto it = graph_.node_map.find(key);
    ICHECK(it != graph_.node_map.end()) << "Cannot find node " << GetRef<ObjectRef>(key);
    IndexedForwardGraph::Node* node = it->second;
    if (node->ref == nullptr) {
      node->ref = key;
      node->index = graph_.post_dfs_order.size();
      graph_.post_dfs_order.push_back(node);
    } else {
      ICHECK(node->ref == key);
    }
  }

  IndexedForwardGraph::Node* CreateNode(const tvm::Object* key) {
    ICHECK(graph_.node_map.find(key) == graph_.node_map.end());
    IndexedForwardGraph::Node* node = arena_->make<IndexedForwardGraph::Node>();
    graph_.node_map[key] = node;
    return node;
  }

 private:
  /*! \brief The whole IRModule */
  const IRModule& mod_;
  /*! \brief Allocator of all the internal node object */
  support::Arena* arena_;
  /*! \brief The output graph */
  IndexedForwardGraph graph_;
  /*! \brief current binding var */
  Optional<Var> cur_binding_var_ = NullOpt;
};

class RelaxFuseMutator : public ExprMutator {
 public:
  // Run the transform
  static IRModule Transform(const IRModule& mod, int fuse_opt_level, size_t max_fuse_depth) {
    // setup the group map.
    RelaxFuseMutator mutator(mod);
    auto graph = IndexedForwardGraphCreator::Create(mod, &mutator.arena_);
    auto groups =
        GraphPartitioner(&mutator.arena_, fuse_opt_level, max_fuse_depth).Partition(graph);
    for (size_t nid = 0; nid < graph.post_dfs_order.size(); ++nid) {
      ICHECK(graph.post_dfs_order[nid]->ref != nullptr);
      mutator.gmap_[graph.post_dfs_order[nid]->ref] = groups[nid];
    }

    // The following line can be used for debug.
    // GroupDebugDumper::Dump(mod, mutator.gmap_);

    for (const auto& kv : mod->functions) {
      Expr func = kv.second;
      const GlobalVar& global_var = kv.first;
      if (func->IsInstance<FunctionNode>()) {
        func = mutator.VisitExpr(func);
        mutator.builder_->AddFuncToContext(Downcast<BaseFunc>(func), global_var->name_hint);
      }
    }

    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit RelaxFuseMutator(const IRModule& mod) : mod_(mod) {}

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) final {
    // Skip Binding Block since it's imprue (with side effect or control flow)
    return GetRef<BindingBlock>(block);
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    const Var& var = binding->var;
    ICHECK(gmap_.count(var.get()));
    cur_group_ = gmap_.at(var.get())->FindRoot();

    auto it = ginfo_.find(cur_group_);
    if (it == ginfo_.end()) {
      // This is a new group
      if (cur_group_->root_ref == var.get()) {
        // Don't create new function if there is only one binding in the new group
        ExprMutator::VisitBinding_(binding);
      } else {
        ginfo_[cur_group_] = GroupInfo();
        builder_->BeginDataflowBlock();
        ExprMutator::VisitBinding_(binding);
      }
    } else {
      Expr new_value = this->VisitExpr(binding->value);
      if (cur_group_->root_ref == var.get()) {
        builder_->EmitOutput(new_value);
        Var new_var = builder_->Emit(MakeNewFunction());
        this->var_remap_[var->vid] = new_var;
      } else {
        builder_->Emit(new_value);
      }
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");

    if (call->op.as<OpNode>()) {
      if (call->op == call_tir_op_) {
        return VisitCallTIR(call);
      } else {
        LOG(FATAL) << "Unsupported OpNode: " << call->op;
        return Expr();
      }
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }

  Expr VisitExpr_(const TupleGetItemNode* op) final {
    if (ginfo_.find(cur_group_) != ginfo_.end()) {
      auto t = ginfo_[cur_group_].GetOrAllocParam(op->tuple);
      return TupleGetItem(t, op->index, op->span);
    }
    return ExprMutator::VisitExpr_(op);
  }

  Call VisitCallTIR(const CallNode* call) {
    // Update fused func name
    GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
    BaseFunc prim_func = mod_->Lookup(gv);
    GlobalVar new_gv = this->builder_->AddFuncToContext(prim_func, gv->name_hint);

    // Update fused func arguments
    Tuple call_tir_args = Downcast<Tuple>(call->args[1]);
    if (ginfo_.find(cur_group_) == ginfo_.end()) {
      // No need to make new relax function, direct update call_tir
      call_tir_args = Downcast<Tuple>(this->VisitExpr(call_tir_args));
    } else {
      call_tir_args = GetNewArguments(call_tir_args);

      // Do not move this line outside the branch
      // since c++ set will automatically insert a new element when accessing
      ginfo_[cur_group_].name_hint = ginfo_[cur_group_].name_hint + "_" + gv->name_hint;
    }

    // Create new call
    Array<Expr> args = {new_gv, call_tir_args, call->args[2]};
    return Call(call->op, args, {}, call->type_args, call->span);
  }

  Tuple GetNewArguments(const Tuple& args) {
    Array<Expr> new_args;
    for (Expr arg : args->fields) {
      ICHECK(gmap_.count(arg.get()));
      auto* arg_group = gmap_.at(arg.get())->FindRoot();
      arg = VisitExpr(arg);
      if (cur_group_ != arg_group && arg->IsInstance<VarNode>()) {
        Var param = ginfo_[cur_group_].GetOrAllocParam(arg);
        new_args.push_back(param);
      } else {
        new_args.push_back(arg);
      }
    }
    return Tuple(new_args);
  }

  Expr MakeNewFunction() {
    const GroupInfo& ginfo = ginfo_[cur_group_];
    DataflowBlock block = Downcast<DataflowBlock>(builder_->EndBlock());
    Optional<Expr> output_body;

    for (const relax::Binding& binding : block->bindings) {
      Var var;
      if (const relax::VarBindingNode* var_binding = binding.as<relax::VarBindingNode>()) {
        var = var_binding->var;
      } else if (const relax::MatchShapeNode* shape_binding = binding.as<relax::MatchShapeNode>()) {
        var = shape_binding->var;
      }
      if (var.defined() && !var.as<relax::DataflowVarNode>()) {
        ICHECK(!output_body) << "Only one output is allowed";
        output_body = var;
      }
    }
    ICHECK(output_body) << "There should be at least one output.";
    const Expr& body = output_body.value();
    auto func = Function(NullOpt, ginfo.params, SeqExpr({block}, body), body->checked_type());
    func->shape_ = body->shape_;
    GlobalVar gv = builder_->AddFuncToContext(func, ginfo.name_hint);
    return Call(gv, ginfo.arguments);
  }

 private:
  // Debug function, dump the group assignment in text.
  class GroupDebugDumper : public ExprVisitor {
   public:
    static void Dump(const IRModule& mod,
                     const std::unordered_map<const Object*, GraphPartitioner::Group*>& gmap) {
      GroupDebugDumper dumper(gmap);
      for (const auto& kv : mod->functions) {
        if (const auto* func = kv.second.as<FunctionNode>()) {
          for (const Var& pram : func->params) {
            // Skip function params since they are always a single group
            dumper.skip_objects_.insert(pram.get());
          }
          dumper(GetRef<Expr>(func));
        }
      }

      LOG(INFO) << "Group partition results:\n" << dumper.os_.str();
    }

   private:
    explicit GroupDebugDumper(
        const std::unordered_map<const Object*, GraphPartitioner::Group*>& gmap)
        : gmap_(gmap) {}

    void TryPrintGroup(const ObjectRef& object) {
      if (object->IsInstance<ConstantNode>()) return;
      if (skip_objects_.count(object.get())) return;
      auto it = gmap_.find(object.get());
      if (it == gmap_.end()) return;
      const GraphPartitioner::Group* group = it->second->FindRoot();
      if (const auto* var = object.as<VarNode>()) {
        os_ << var->name_hint();
      } else {
        os_ << object;
      }
      os_ << "(" << object.get() << ")";
      auto g_it = group_id_.find(group);
      os_ << ": Group #";
      if (g_it == group_id_.end()) {
        os_ << (group_id_[group] = group_id_.size());
      } else {
        os_ << g_it->second;
      }
      os_ << "\n";

      // Prevent showing the same node multiple times
      skip_objects_.insert(object.get());
    }

    void VisitExpr(const Expr& expr) final {
      TryPrintGroup(expr);
      ExprVisitor::VisitExpr(expr);
    }

   private:
    std::unordered_map<const GraphPartitioner::Group*, size_t> group_id_;
    std::ostringstream os_;
    const std::unordered_map<const Object*, GraphPartitioner::Group*>& gmap_;
    std::unordered_set<const Object*> skip_objects_;
  };

 private:
  struct GroupInfo {
    // The parameters of the function.
    Array<Var> params;
    // The arguments to call the functions.
    Array<Expr> arguments;
    // The name hint for the group
    String name_hint = "fused";

    Var GetOrAllocParam(const Expr& arg) {
      // run linear scan as most fused groups contain only a few inputs.
      for (size_t i = 0; i < arguments.size(); ++i) {
        if (arg.same_as(arguments[i])) return params[i];
      }
      // create a new parameter.
      if (const auto* arg_var = arg.as<VarNode>()) {
        params.push_back(Var(arg_var->name_hint(), arg_var->shape(), arg_var->checked_type_));
      } else {
        // TODO(siyuan): need enhance it.
        LOG(FATAL) << "ValueError: call args must be a var for now.";
      }
      arguments.push_back(arg);
      return params.back();
    }
  };
  /*! \brief Internal arena. */
  support::Arena arena_;
  /*! \brief The group assignment map. */
  std::unordered_map<const Object*, GraphPartitioner::Group*> gmap_;
  /*! \brief Internal group information map. */
  std::unordered_map<GraphPartitioner::Group*, GroupInfo> ginfo_;
  /*! \brief The IRModule. */
  IRModule mod_;
  /*! \brief The current group. */
  GraphPartitioner::Group* cur_group_;
};

class TIRFuseMutator : public ExprMutator {
 public:
  static IRModule Transform(const IRModule& mod) {
    TIRFuseMutator mutator(mod);

    BaseFunc main_func = mod->Lookup("main");
    mutator.func_info_ = FuseFuncInfo("main", false);
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
      // Emit primitive relax Function
      GlobalVar gv = Downcast<GlobalVar>(call->op);
      BaseFunc func = mod_->Lookup(gv);
      FuseFuncInfo info = func_info_;
      this->func_info_ = FuseFuncInfo(gv->name_hint, true);
      GlobalVar new_gv =
          this->builder_->AddFuncToContext(Downcast<BaseFunc>(VisitExpr(func)), gv->name_hint);
      func_info_ = info;
      return Call(new_gv, call->args, call->attrs, call->type_args, call->span);
    }

    if (call->op != call_tir_op_) {
      return e;
    }

    GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
    tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(gv));

    if (func_info_.is_primitive) {
      func_info_.prim_funcs.push_back(func);
      // update func_info_.param_map
      const Array<Expr> call_tir_args = Downcast<Tuple>(call->args[1])->fields;
      for (size_t i = 0; i < call_tir_args.size(); ++i) {
        if (call_tir_args[i]->IsInstance<ConstantNode>()) {
          func_info_.arguments.push_back(call_tir_args[i]);
        } else if (call_tir_args[i]->IsInstance<VarNode>()) {
          Var arg_var = Downcast<Var>(call_tir_args[i]);
          auto it = func_info_.var2param.find(arg_var);
          if (it == func_info_.var2param.end()) {
            // add it to the arg list if the arg is not the result of previous call_tir
            func_info_.arguments.push_back(arg_var);
          } else {
            const tir::Var& producer_param = Downcast<tir::Var>((*it).second);
            const tir::Var& consumer_param = func->params[i];
            func_info_.param_map.Set(consumer_param, producer_param);
          }
        } else {
          ICHECK(false) << "Only var and constant are allowed";
        }
      }
      return e;
    } else {
      GlobalVar new_gv = this->builder_->AddFuncToContext(func, gv->name_hint);
      return Call(call->op, {new_gv, call->args[1], call->args[2]}, call->attrs, call->type_args,
                  call->span);
    }
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    if (!func_info_.is_primitive) {
      return ExprMutator::VisitBinding_(binding);
    }
    Expr value = this->VisitExpr(binding->value);
    if (const auto* call = value.as<CallNode>()) {
      if (!binding->var->IsInstance<DataflowVarNode>()) {
        // Emit call_tir func if it's the output call
        tir::PrimFunc func = tir::FusePrimFuncs(func_info_.prim_funcs, func_info_.param_map);
        GlobalVar gv = this->builder_->AddFuncToContext(func, func_info_.name_hint);
        Array<Expr> call_args = {gv, Tuple(func_info_.arguments), call->args[2]};
        Call new_call_tir(call_tir_op_, call_args, call->attrs, call->type_args);
        Var output = this->builder_->EmitOutput(new_call_tir);
        this->var_remap_[binding->var->vid] = output;
      } else {
        // Update func_info_.var2param
        GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
        tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
        const Expr& output_shapes = call->args[2];
        if (const auto* tuple_output_shapes = output_shapes.as<TupleNode>()) {
          // set var2param to a array if there is more than one output.
          size_t output_size = tuple_output_shapes->fields.size();
          Array<tir::Var> output_param(func->params.end() - output_size, func->params.end());
          func_info_.var2param.Set(binding->var, output_param);
        } else {
          func_info_.var2param.Set(binding->var, func->params.back());
        }
      }
    } else if (const auto* tuple_get_item = value.as<TupleGetItemNode>()) {
      ICHECK(binding->var->IsInstance<DataflowVarNode>())
          << "Currently TupleGetItem outputs are not allowed";
      const Var& tuple_var = Downcast<Var>(tuple_get_item->tuple);
      auto it = func_info_.var2param.find(tuple_var);
      if (it == func_info_.var2param.end()) {
        // Directly emit tuple if it's extern input
        Var lv = this->builder_->Emit(value);
        this->var_remap_[binding->var->vid] = lv;
      } else {
        // Update var2param if the input is local var
        Array<tir::Var> params = Downcast<Array<tir::Var>>((*it).second);
        func_info_.var2param.Set(binding->var, params[tuple_get_item->index]);
      }
    } else {
      LOG(FATAL) << "Unsupported binding value: " << value;
    }
  }

 private:
  struct FuseFuncInfo {
    FuseFuncInfo() = default;
    FuseFuncInfo(const String& name_hint, bool is_primitive)
        : name_hint(name_hint), is_primitive(is_primitive) {}

    /*! \brief The relax function name hint */
    String name_hint = "";
    /*! \brief An boolean indicate if the function if to be fused */
    bool is_primitive = false;
    /*! \brief The prim_funcs to be fused. */
    Array<tir::PrimFunc> prim_funcs;
    /*!
     * \brief The mapping from relax to prim_func param
     * \note The rhs can be a tir::Var or Array<tir::Var> (for tuple return)
     */
    Map<Var, ObjectRef> var2param;
    /*!
     * \brief A map indicate how data exchange between functions.
     *        The map is from consumer params to the producer params.
     */
    Map<tir::Var, tir::Var> param_map;
    /*! \brief The arguments for calling prim_func */
    Array<Expr> arguments;
  };

  /*! \brief The IRModule */
  const IRModule& mod_;
  /*! \brief The IRModule */
  FuseFuncInfo func_info_;
};

class Inliner : public ExprMutator {
 public:
  static IRModule Transform(const IRModule& mod) {
    Inliner inliner(mod);
    for (const auto& kv : mod->functions) {
      BaseFunc func = kv.second;
      const GlobalVar& global_var = kv.first;
      // We won't add current PrimFunc to the context
      if (global_var->name_hint == "main") {
        Expr new_func = inliner.VisitExpr(func);
        inliner.builder_->AddFuncToContext(Downcast<BaseFunc>(new_func), global_var->name_hint);
      } else if (func->IsInstance<tir::PrimFuncNode>()) {
        inliner.builder_->AddFuncToContext(func, global_var->name_hint);
      }
    }

    return inliner.builder_->GetContextIRModule();
  }

 private:
  explicit Inliner(const IRModule& mod) : mod_(mod) {}

  void VisitBinding_(const VarBindingNode* binding) final {
    Optional<Function> _relax_func = get_relax_call(binding);
    if (!_relax_func.defined()) {
      return ExprMutator::VisitBinding_(binding);
    }
    Function relax_func = _relax_func.value();
    Call call = Downcast<Call>(binding->value);
    VisitPrimitiveFunc(relax_func, call->args, binding->var);
  }

 private:
  void VisitPrimitiveFunc(const Function& func, const Array<Expr>& args, const Var& binding_var) {
    // update var_remap_ via function params
    ICHECK_EQ(func->params.size(), args.size());
    for (size_t i = 0; i < func->params.size(); ++i) {
      const Var& param = func->params[i];
      const Expr& arg = args[i];
      this->var_remap_[param->vid] = Downcast<Var>(arg);
    }

    const auto* seq = func->body.as<SeqExprNode>();
    ICHECK(seq != nullptr);
    ICHECK_EQ(seq->blocks.size(), 1);
    const BindingBlock& block = seq->blocks[0];
    ICHECK(block->IsInstance<DataflowBlockNode>());
    for (const Binding& binding : block->bindings) {
      if (const auto* var_binding = binding.as<VarBindingNode>()) {
        Var var = this->builder_->Emit(VisitExpr(var_binding->value));
        this->var_remap_[var_binding->var->vid] = var;
      } else {
        ICHECK(false) << "Unsupported binding";
      }
    }
    // Update return value remap
    Var return_var = Downcast<Var>(seq->body);
    this->var_remap_[binding_var->vid] = this->var_remap_[return_var->vid];
  }

  Optional<Function> get_relax_call(const VarBindingNode* binding) {
    const auto* call = binding->value.as<CallNode>();
    // Cond 1. binding value is a call node.
    if (call == nullptr) return NullOpt;
    // Cond 2. Call node op must be GlobalVar
    if (!call->op->IsInstance<GlobalVarNode>()) return NullOpt;
    GlobalVar gv = Downcast<GlobalVar>(call->op);
    // Cond 3. The GlobalVar must be in the IRModule
    auto it = mod_->functions.find(gv);
    if (it == mod_->functions.end()) return NullOpt;
    // Cond 4. The function must be a relax function
    const BaseFunc& func = (*it).second;
    if (!func->IsInstance<FunctionNode>()) return NullOpt;
    return Downcast<Function>(func);
  }

 private:
  const IRModule& mod_;
};

IRModule FuseOps(IRModule mod, int opt_level, size_t max_fuse_depth) {
  mod = RelaxFuseMutator::Transform(mod, opt_level, max_fuse_depth);
  // const auto* f = runtime::Registry::Get("script.AsRelaxScript");
  // String s = (*f)(mod, false);
  // std::cout << s << std::endl;
  mod = TIRFuseMutator::Transform(mod);
  mod = Inliner::Transform(mod);
  return mod;
}

namespace transform {

Pass FuseOps(int fuse_opt_level) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) {
        int opt_level = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
        auto max_fuse_depth = pc->GetConfig("relax.FuseOps.max_depth", Integer(kMaxFusedOps));
        return relax::FuseOps(m, opt_level, max_fuse_depth.value());
      };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"FuseOps",      //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.FuseOps").set_body_typed(FuseOps);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

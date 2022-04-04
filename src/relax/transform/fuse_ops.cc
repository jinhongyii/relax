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
 * \file src/relax/transform/fuse_ops.cc
 * \brief This file contains a pass which groups bindings in a dataflow block of Relax
 * functions and generate a new grouped Relax function for each group, according to the fusion
 * algorithm described below. By grouping bindings into new Relax functions, we substitute the
 * bindings in the function being manipulated into function calls to the new grouped function.
 *
 * A follow-up pass named "FuseTIR" will generate a TIR PrimFunc for each grouped function.
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

class GraphCreator : public ExprVisitor {
 public:
  /*!
   * \brief Create a IndexedForwardGraph according to the input module. The graph will be used for
   * graph partition and operator fusion.
   * \param mod The module which the creation accords to
   * \param arena The allocator of all the internal node objects
   * \return The created IndexedForwardGraph
   */
  static IndexedForwardGraph Create(IRModule mod, support::Arena* arena) {
    // Since cross-function call is not supported yet, FuseOps only serves the entry function, whose
    // name is "main".
    auto relax_func = Downcast<Function>(mod->Lookup("main"));
    GraphCreator creator(mod, arena);
    creator(relax_func);

    // The algorithm of the graph creator ensures that each created node will be added to the
    // post-dfs order and will be set its op pattern. Thus we check whether all these containers
    // have the same size.
    size_t n_nodes = creator.graph_.node_map.size();
    ICHECK_EQ(n_nodes, creator.graph_.post_dfs_order.size());
    ICHECK_EQ(n_nodes, creator.initialized_nodes_.size());

    return creator.graph_;
  }

 private:
  explicit GraphCreator(IRModule mod, support::Arena* arena)
      : mod_(std::move(mod)), arena_(arena) {}

  void VisitExpr_(const FunctionNode* func) final {
    for (const Var& param : func->params) {
      IndexedForwardGraph::Node* param_node = CreateNode(param.get());
      LOG(INFO) << "[2]. Create node for param " << param << ". node is " << param_node;
      // The parameter is passed in from the outside, and thus it's marked as an external reference,
      // and it's pattern is `kOpaque`.
      MarkAsExternRef(param_node);
      SetNodePattern(param_node, OpPatternKind::kOpaque);
      AddToPostDFSOrder(param_node, param.get());
    }
    ExprVisitor::VisitExpr_(func);
  }

  void VisitBindingBlock(const BindingBlock& block) final {
    if (const auto* df_block = block.as<DataflowBlockNode>()) {
      VisitBindingBlock_(df_block);
    }
    // We skip ordinary binding blocks since they might be impure (with side effect or control flow)
  }

  // TODO(tvm-team): how to deal with MatchShape binding here

  void VisitBinding_(const VarBindingNode* binding) final {
    CHECK(cur_binding_var_node_ == nullptr)
        << "We are visiting a new VarBinding inside an outer binding, which is not allowed";
    cur_binding_var_node_ = CreateNode(binding->var.get());
    LOG(INFO) << "[3]. Create node for binding var " << binding->var->name_hint() << ". node is "
              << cur_binding_var_node_ << ", type key is " << binding->var->GetTypeKey();

    // If the variable is not a dataflow variable, it must be the output variable of this dataflow
    // block
    if (!binding->var->IsInstance<DataflowVarNode>()) {
      this->MarkAsExternRef(cur_binding_var_node_);
    }
    ExprVisitor::VisitBinding_(binding);
    AddToPostDFSOrder(cur_binding_var_node_, binding->var.get());
    cur_binding_var_node_ = nullptr;
  }

  void VisitExpr(const Expr& expr) final {
    if (cur_binding_var_node_ == nullptr) {
      // Case 1. The expression is not under a binding. No action is needed.
    } else if (expr->IsInstance<CallNode>() || expr->IsInstance<TupleGetItemNode>()) {
      // Case 2. The type of the expression is supported by fusion (as defined below). No action is
      // needed again - we will recurse into this expression and let the visitor deal with such
      // expressions.
    } else if (!IsLeaf(expr)) {
      // Case 3. The type of the expression is not fusion-supported and the expression is not a
      // leaf. In this case, we set the pattern of the current binding variable to be `kOpaque`.
      ICHECK(cur_pattern_ == OpPatternKind::kOpaque);
      SetNodePattern(cur_binding_var_node_, OpPatternKind::kOpaque);
    } else {
      // Case 4. The expression is a leaf expression, which currently is not fusion-supported.
      //   - Under such circumstances, if the current binding value is exactly the expression
      //   itself, the pattern of the current binding variable is not set.
      //   - Otherwise, the pattern of the current binding variable must have been set.
      //   - Thus, we set the pattern of the current binding variable to `kOpaque` (since leaf
      //   expressions are not fusion-supported) if it hasn't been set yet.
      if (initialized_nodes_.find(cur_binding_var_node_) == initialized_nodes_.end()) {
        ICHECK(cur_pattern_ == OpPatternKind::kOpaque);
        SetNodePattern(cur_binding_var_node_, OpPatternKind::kOpaque);
      }
    }
    ExprVisitor::VisitExpr(expr);
  }

  /********** Non-Leaf Expression Nodes **********/

  void VisitExpr_(const CallNode* call) final {
    // If the function call is not under a binding, there is no need to recurse into it.
    if (cur_binding_var_node_ == nullptr) {
      return;
    }
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");

    // - If the op being called is a TIR PrimFunc, we get the function op pattern directly from the
    // function attribute and visit the arguments one by one.
    // - Otherwise, the pattern of the current binding variable node is set to `kOpaque`, and we
    // recurse into the call expression.
    const auto* op = call->op.as<OpNode>();
    if (op == call_tir_op_.get()) {
      const GlobalVar& global_var = Downcast<GlobalVar>(call->args[0]);
      tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(global_var));

      const Tuple& args = Downcast<Tuple>(call->args[1]);
      // TODO(tvm-team): handle the shape argument (args[3])
      Optional<Integer> opt_pattern = func->GetAttr<Integer>("op_pattern");
      OpPatternKind pattern;
      if (opt_pattern.defined()) {
        pattern = static_cast<OpPatternKind>(Downcast<IntImm>(opt_pattern)->value);
      } else {
        pattern = OpPatternKind::kOpaque;
      }
      // The pattern of the current binding variable node is set to the pattern of this operator.
      SetNodePattern(cur_binding_var_node_, pattern);

      for (const Expr& arg : args->fields) {
        // If the operator pattern was detected to be `kBroadcast`, and meanwhile this argument has
        // the same shape as the operator output, the relation between this argument and the output
        // is actually element-wise. And in this case we change the pattern to `kElemWise`
        // temporarily.
        if (pattern == OpPatternKind::kBroadcast && structural_equal_(call->shape_, arg->shape_)) {
          LOG(INFO) << "[13]. Change current pattern to element-wise";
          cur_pattern_ = OpPatternKind::kElemWise;
        } else {
          cur_pattern_ = pattern;
        }
        VisitExpr(arg);
      }
    } else {
      SetNodePattern(cur_binding_var_node_, OpPatternKind::kOpaque);
      ExprVisitor::VisitExpr_(call);
    }

    // Restore the value of `cur_pattern_`.
    cur_pattern_ = OpPatternKind::kOpaque;
  }

  void VisitExpr_(const TupleGetItemNode* tuple_item) final {
    // If the tuple-get-item node is not under a binding, there is no need to recurse into it.
    if (cur_binding_var_node_ == nullptr) {
      return;
    }

    SetNodePattern(cur_binding_var_node_, OpPatternKind::kInjective);
    cur_pattern_ = OpPatternKind::kInjective;
    VisitExpr(tuple_item->tuple);

    // Restore the value of `cur_pattern_`.
    cur_pattern_ = OpPatternKind::kOpaque;
  }

  /********** Leaf Expression Nodes **********/

  void VisitExpr_(const ConstantNode* constant) final {
    // If the constant is not under a binding, there is no need to recurse into it.
    if (cur_binding_var_node_ == nullptr) {
      return;
    }
    // TODO(tvm-team): what about constant shape in match-shape?

    // If we're visiting the constant for the first time, we create a node for it.
    // Otherwise, we fetch the node from the node map of the graph.
    auto it_const = graph_.node_map.find(constant);
    IndexedForwardGraph::Node* const_node = nullptr;
    if (it_const == graph_.node_map.end()) {
      const_node = CreateNode(constant);
      LOG(INFO) << "[4]. Create node for constant. node is " << const_node;
      // Since we never fuse constants, the pattern of the constant is set to `kOpaque`.
      SetNodePattern(const_node, OpPatternKind::kOpaque);
      AddToPostDFSOrder(const_node, constant);
    } else {
      const_node = it_const->second;
    }
    AddEdge(const_node, cur_binding_var_node_, OpPatternKind::kOpaque);
  }

  void VisitExpr_(const VarNode* var) final {
    // If the variable is not under a binding, there is no need to recurse into it.
    if (cur_binding_var_node_ == nullptr) {
      return;
    }
    auto it_var = graph_.node_map.find(var);
    CHECK(it_var != graph_.node_map.end()) << "The variable is supposed to be defined before";

    // - If the variable is a component of some other binding value (call or tuple-get-item),
    // `cur_pattern_` is supposed to be properly set already.
    // - Otherwise, `cur_pattern_` is supposed to be `kOpaque` by default.
    AddEdge(it_var->second, cur_binding_var_node_, cur_pattern_);
  }

  void VisitExpr_(const DataflowVarNode* var) final { VisitExpr_(GetRef<Var>(var).get()); }

  /********** Helper Functions **********/

  /*!
   * \brief Check whether the expression is a leaf expression
   * \param expr The expression to be checked
   * \return Whether the expression is a leaf expression
   * \note In order to avoid too much refactor, this method is a simple copy-paste of the is-leaf
   * check in "block_builder.cc". And it should be refactored in the future.
   * \sa src/relax/ir/block_builder.cc
   */
  static bool IsLeaf(const Expr& expr) {
    // NOTE: Tuples are treated as leaf nodes for ergonomics
    return expr.as<VarNode>() || expr.as<GlobalVarNode>() || expr.as<ConstantNode>() ||
           expr.as<ShapeExprNode>() || expr.as<ExternFuncNode>() || expr.as<OpNode>() ||
           expr.as<TupleNode>();
  }

  /*!
   * \brief Create a graph node corresponding to the input key
   * \param key The object which is used to create the graph node
   * \return The created graph node
   * \note The node corresponding to each key is supposed to be created for only once
   */
  IndexedForwardGraph::Node* CreateNode(const Object* key) {
    ICHECK(graph_.node_map.find(key) == graph_.node_map.end())
        << "The node corresponding to the input key is not supposed to be created before";
    auto* node = arena_->make<IndexedForwardGraph::Node>();
    graph_.node_map[key] = node;
    return node;
  }

  /*!
   * \brief Append the input node to the post-dfs order of the graph
   * \param node The node to be appended
   * \param key The key corresponding to the node
   * \note Each node is supposed to be appended to the post-dfs order for only once
   */
  void AddToPostDFSOrder(IndexedForwardGraph::Node* node, const Object* key) {
    auto it = graph_.node_map.find(key);
    ICHECK(it != graph_.node_map.end() && it->second == node)
        << "The node must have been created before adding to the post-dfs order";

    // We only set the reference of the node when adding it to the post-dfs order. Thus, if the
    // reference of a node is already set, it must have been appended to the post-dfs order.
    ICHECK(node->ref == nullptr)
        << "The node is not supposed to be added into the post-dfs order before";

    LOG(INFO) << "[6]. Add node " << node << " to post-dfs order";
    node->ref = key;
    node->index = graph_.post_dfs_order.size();
    graph_.post_dfs_order.push_back(node);
  }

  /*!
   * \brief Add an edge from the input start to the input end in the graph, with specific pattern
   * \param start The start of the edge
   * \param end The end of the edge
   * \param pattern The pattern of this edge
   */
  void AddEdge(IndexedForwardGraph::Node* start, IndexedForwardGraph::Node* end,
               OpPatternKind pattern) {
    LOG(INFO) << "[8]. Edge: " << start << " ---> " << end << ", pattern: " << (int)pattern;
    auto* link = arena_->make<LinkNode<IndexedForwardGraph::Edge>>();
    link->value.node = end;
    link->value.pattern = pattern;
    start->outputs.Push(link);
  }

  /*!
   * \brief Mark a given node as "external reference", which means the node cannot be fused as an
   * intermediate node
   * \param node The graph node to be marked
   */
  void MarkAsExternRef(IndexedForwardGraph::Node* node) { node->extern_ref = true; }

  /*!
   * \brief Set the pattern of the input node
   * \param node The graph node to be set
   * \param pattern The pattern of the node
   */
  void SetNodePattern(IndexedForwardGraph::Node* node, OpPatternKind pattern) {
    ICHECK(initialized_nodes_.find(node) == initialized_nodes_.end())
        << "The input node is supposed to be set pattern for only once";
    LOG(INFO) << "[7]. Set pattern of node " << node << " to " << static_cast<int>(pattern);
    initialized_nodes_.insert(node);
    node->pattern = pattern;
  }

 private:
  /*! \brief The IRModule from which the indexed forward graph is created */
  IRModule mod_;
  /*! \brief The allocator of all the internal node objects */
  support::Arena* arena_;
  /*! \brief The created indexed forward graph */
  IndexedForwardGraph graph_;
  /*! \brief The variable in the current VarBinding */
  IndexedForwardGraph::Node* cur_binding_var_node_ = nullptr;
  /*! \brief The op pattern of the current binding */
  OpPatternKind cur_pattern_ = OpPatternKind::kOpaque;
  /*! \brief The graph nodes whose patterns are set */
  std::unordered_set<IndexedForwardGraph::Node*> initialized_nodes_;
  /*! \brief The structural equality checker */
  StructuralEqual structural_equal_;
};

/*!
 * \brief The ExprMutator used to create a new grouped function
 * \details The workflow of this ExprMutator is:
 *  - The bindings in the function will be added by OperatorFusor via `AppendBinding(...)`.
 *  - When adding a new binding through `AppendBinding(...)`, we check whether the variables and
 *  constants used by the binding are defined by some previous added binding. And for the undefined
 *  variables and constants, we add them to the argument list and created new variables as the
 *  corresponding parameters.
 *  - When `CreateFunction()` is called, we go through each binding and update the binding with the
 *  new parameters. After that we wrap all bindings with a DataflowBlock and a Function.
 */
class FunctionCreator : public ExprMutator {
 public:
  /*!
   * \brief Append a new binding to this function and possibly create new parameters for the
   * function accordingly
   * \param binding The binding to be appended
   * \note Allowed bindings are:
   *  - VarBinding with value being a call node calling `relax.call_tir`.
   *  - VarBinding with value being a tuple-get-item node.
   * // TODO(tvm-team): handle match shape
   */
  void AppendBinding(const Binding& binding) {
    ICHECK(!function_.defined())
        << "The `function_` is supposed to be uncreated when adding bindings";
    ICHECK(!has_output_var_)
        << "It's not allowed to append more bindings once the function has an output variable";

    if (const auto* var_binding = binding.as<VarBindingNode>()) {
      if (const auto* call = var_binding->value.as<CallNode>()) {
        ICHECK(call->op == Op::Get("relax.call_tir"));
        const GlobalVar& global_var = Downcast<GlobalVar>(call->args[0]);
        // Update the name of the function.
        name_ = name_ + "_" + Downcast<GlobalVar>(call->args[0])->name_hint;

        const Tuple& args = Downcast<Tuple>(call->args[1]);
        for (const Expr& arg : args->fields) {
          CheckDefAndUpdateParam(arg);
        }
        // TODO(tvm-team): handle shape expr
      } else {
        const auto* tuple_item = var_binding->value.as<TupleGetItemNode>();
        ICHECK(tuple_item != nullptr);
        CheckDefAndUpdateParam(tuple_item->tuple);
      }

      // Mark the binding variable as defined.
      defined_vars_.insert(var_binding->var.get());
      // Set `has_output_var_` to true if the binding variable is an output variable (a.k.a. is not
      // a dataflow variable).
      if (!var_binding->var->IsInstance<DataflowVarNode>()) {
        has_output_var_ = true;
      }
    } else {
      // TODO(tvm-team): handle match_shape
    }
    bindings_.push_back(binding);
  }

  /*!
   * \brief Create the grouped function according according to the collected bindings and parameters
   * \note The created function won't be returned immediately. Tt's stored in the `function_` field.
   */
  void CreateFunction() {
    // Step 1. Start constructing a new dataflow block.
    builder_->BeginDataflowBlock();
    // Step 2. Visit each binding, except the last one, one by one.
    for (int i = 0; i < static_cast<int>(bindings_.size()) - 1; ++i) {
      VisitBinding(bindings_[i]);
    }

    // Step 3. Since the binding var of the last binding should be an output variable, we deal with
    // the last binding separately.
    const auto* last_binding = bindings_.back().as<VarBindingNode>();
    ICHECK(last_binding != nullptr) << "The last binding of a group is supposed to be a VarBinding";
    Expr binding_value = VisitExpr(last_binding->value);
    Var output_var(last_binding->var->vid, NullOpt, last_binding->var->checked_type_);
    output_var->shape_ = last_binding->var->shape_;
    builder_->EmitOutput(VarBinding(output_var, binding_value));

    // Step 4. Finish constructing the new block.
    BindingBlock new_block = builder_->EndBlock();
    // Step 5. Create a new global variable and the function.
    global_var_ = GlobalVar(name_);
    function_ = Function(/*name=*/global_var_,                       //
                         /*params=*/params_,                         //
                         /*body=*/SeqExpr({new_block}, output_var),  //
                         /*ret_type=*/output_var->checked_type_);
    function_->shape_ = output_var->shape_;
  }

  /*! \brief The original bindings of the function */
  Array<Binding> bindings_;
  /*! \brief The parameters of the function */
  Array<Var> params_;
  /*! \brief The arguments to call the function on the caller side */
  Array<Expr> arguments_;
  /*! \brief The name for the fused function */
  String name_ = "fused";
  /*! \brief The global variable corresponding to the constructed function */
  GlobalVar global_var_;
  /*! \brief The constructed Relax function */
  Function function_{nullptr};

 private:
  /*!
   * \brief Check whether the input expression is defined within this function. If not, create a new
   * parameter for the expression.
   * \param expr The expression to be checked
   */
  void CheckDefAndUpdateParam(const Expr& expr) {
    // If the expression has already served as an argument, no need to create another one for it.
    auto it = std::find(arguments_.begin(), arguments_.end(), expr);
    if (it != arguments_.end()) {
      return;
    }

    // If the expression is not a variable or is a undefined variable, it should be populated as a
    // parameter of the relax function.
    const auto* var = expr.as<VarNode>();
    if (var == nullptr || defined_vars_.count(var) == 0) {
      String name{nullptr};
      if (var != nullptr) {
        name = var->name_hint();
      } else {
        name = String("param_" + std::to_string(n_param_for_const_++));
      }

      Var param(std::move(name),               //
                /*shape_annotation=*/NullOpt,  //
                /*type_annotation=*/expr->checked_type_);
      param->shape_ = expr->shape_;
      arguments_.push_back(expr);
      params_.push_back(param);
    }
  }

  Expr VisitExpr(const Expr& expr) final {
    // If the expression serves as an argument, return its correspondng parameter.
    auto it = std::find(arguments_.begin(), arguments_.end(), expr);
    if (it != arguments_.end()) {
      return params_[it - arguments_.begin()];
    }
    // Otherwise, recurse into this expression.
    return ExprMutator::VisitExpr(expr);
  }

 private:
  /*! \brief The variables defined in this function */
  std::unordered_set<const VarNode*> defined_vars_;
  /*! \brief The number of parameters reserved for constants */
  int n_param_for_const_ = 0;
  /*! \brief The boolean indicating whether the input bindings have an output variable */
  bool has_output_var_ = false;
};

/*!
 * \brief The ExprMutator used to fuse the operators in Relax functions
 * \details Given the partition results on the indexed-forward graph, for each group whose size is
 * larger than one, we create a new grouped function for it, containing all bindings in that group.
 * And we substitute the bindings in a group with a single function call to the newly created
 * grouped function. The workflow of this ExprMutator is: for each dataflow block,
 *   - we go through the bindings one by one. For each binding, if it is in a group whose size is
 *   larger than one, we add the binding to the function of the group it is in and update the
 *   parameters and arguments of that function;
 *   - then we finalize all the grouped functions by updating their bindings using BlockBuilder;
 *   - lastly, we go through the bindings again and substitute the bindings in a group with a single
 *   call to the corresponding grouped function.
 *
 * After transforming a Relax function, we update the function in the IRModule. Besides, we add all
 * newly created grouped function to the IRModule.
 */
class OperatorFusor : public ExprMutator {
 public:
  /*!
   * \brief Construct a new operator fusor. Given the indexed-forward graph and the graph partition
   * result on that graph, the constructor creates a mapping from each leaf AST object
   * (e.g. parameters, variables, constants) to the group of the node corresponding to the object
   * in the graph.
   * \param mod The IRModule to be transformed
   * \param graph The indexed-forward graph of the input IRModule
   * \param groups The grouped result of the group partition on the input indexed-forward graph.
   */
  explicit OperatorFusor(IRModule mod, const IndexedForwardGraph& graph,
                         const std::vector<GraphPartitioner::Group*>& groups)
      : mod_(std::move(mod)) {
    for (int nid = 0; nid < static_cast<int>(graph.post_dfs_order.size()); ++nid) {
      GraphPartitioner::Group* group_root = groups[nid]->FindRoot();
      ICHECK(group_root != nullptr);
      ICHECK(graph.post_dfs_order[nid]->ref != nullptr);
      obj2group_[graph.post_dfs_order[nid]->ref] = group_root;
    }
  }

  /*!
   * \brief The main transformation on the IRModule
   * \return The new IRModule after transformation
   */
  IRModule Transform() {
    // Step 1. Fetch the main function and apply transformation by recursing into the function.
    //   - Since cross-function call is not supported yet, FuseOps only serves the entry function,
    //   whose name is "main".
    GlobalVar main_gv = mod_->GetGlobalVar("main");
    auto main_func = Downcast<Function>(mod_->Lookup("main"));
    auto updated_main_func = Downcast<Function>(VisitExpr(main_func));

    // Step 2. Update the main function in the IRModule.
    IRModuleNode* p_mod = mod_.CopyOnWrite();
    p_mod->Update(main_gv, updated_main_func);

    // Step 3. Add the new functions into the IRModule.
    for (const auto& kv : new_functions_) {
      p_mod->Add(kv.second.first, kv.second.second);
    }

    return GetRef<IRModule>(p_mod);
  }

 private:
  BindingBlock VisitBindingBlock(const BindingBlock& block) final {
    if (const auto* df_block = block.as<DataflowBlockNode>()) {
      return VisitBindingBlock_(df_block);
    }
    // We skip ordinary binding blocks since they might be impure (with side effect or control flow)
    return block;
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) final {
    group2func_.clear();

    // Step 1. Collect the bindings for each grouped function.
    CollectFuncBindings(block->bindings);

    // Step 2. Create the grouped function for each group.
    for (auto& kv : group2func_) {
      FunctionCreator& creator = kv.second;
      creator.CreateFunction();
    }

    // Step 3. Start generating the new binding block.
    //  - For groups with single binding, we directly recurse into the binding and emit the new one.
    //  - For groups with multiple bindings, we emit the call to the grouped function only when
    //  visiting the last binding of the group, because only by doing this we don't break the
    //  dependencies among the bindings of different groups. And therefore, we will skip all but the
    //  last binding of the group.
    builder_->BeginDataflowBlock();
    for (int i = 0; i < static_cast<int>(block->bindings.size()); ++i) {
      const Binding& binding = block->bindings[i];

      // Case 1. If the binding is the only binding in its group, recurse into it and emit the
      // transformed binding as usual.
      GraphPartitioner::Group* group = GetGroupFromBinding(binding);
      if (group->num_nodes == 1) {
        VisitBinding(binding);
        continue;
      }

      const auto& it_creator = group2func_.find(group);
      ICHECK(it_creator != group2func_.end());
      const FunctionCreator& func_info = it_creator->second;

      // Case 2. If the binding is not the last binding of the group, we skip it.
      if (!func_info.bindings_.back().same_as(binding)) {
        continue;
      }

      // Case 3. The binding is the last binding of the group.
      const auto* var_binding = binding.as<VarBindingNode>();
      ICHECK(var_binding != nullptr) << "The last binding of a group whose size is larger than 1 "
                                        "is supposed to be a variable binding";

      // Step a. Add the grouped function of this group to the field `new_functions_` of this fusor
      // with deduplication.
      std::pair<GlobalVar, Function> gv_func_pair =
          AddFuncWithDeduplication(func_info.global_var_, func_info.function_);
      new_functions_[gv_func_pair.first->name_hint] = gv_func_pair;

      // Step b. Create the call to the deduplicated function, and then emit the call.
      //  - If this binding is the last binding of the current binding block, emit an output
      //  variable.
      //  - Otherwise, emit a dataflow variable.
      Var new_var{nullptr};
      Call call_to_emit = Call(gv_func_pair.first, UpdateArgs(func_info.arguments_));
      if (i < static_cast<int>(block->bindings.size()) - 1) {
        new_var = builder_->Emit(call_to_emit);
      } else {
        new_var = builder_->EmitOutput(call_to_emit);
      }

      // Step c. Update the mapping used for the remapping of the binding variables.
      var_remap_[var_binding->var->vid] = new_var;
    }
    // Step 4. Finish the binding block generation.
    return builder_->EndBlock();
  }

  /*!
   * \brief Collect the bindings for each grouped function and update the information of the grouped
   * function
   * \param bindings The bindings to be collected
   * \note The function update is done by `AppendBinding(...)`
   */
  void CollectFuncBindings(const Array<Binding>& bindings) {
    for (const Binding& binding : bindings) {
      // If the binding is the only binding in its group, there is no need to create a new function.
      GraphPartitioner::Group* group = GetGroupFromBinding(binding);
      if (group->num_nodes == 1) {
        continue;
      }
      // Add the binding to the grouped function it's in, and update the function information
      // accordingly.
      FunctionCreator& func_info = group2func_[group];
      func_info.AppendBinding(binding);
    }
  }

  /*!
   * \brief Get the group which the input binding is in
   * \param binding The binding to be queried
   * \return The pointer to the group which the input binding is in
   */
  GraphPartitioner::Group* GetGroupFromBinding(const Binding& binding) {
    Var var{nullptr};
    if (const auto* var_binding = binding.as<VarBindingNode>()) {
      var = var_binding->var;
    } else {
      const auto* match_shape = binding.as<MatchShapeNode>();
      ICHECK(match_shape != nullptr);
      var = match_shape->var;
    }

    const auto& it_group = obj2group_.find(var.get());
    ICHECK(it_group != obj2group_.end());
    GraphPartitioner::Group* group = it_group->second;
    ICHECK(group->FindRoot() == group);
    return group;
  }

  /*!
   * \brief Update the pre-stored arguments according to the variable remapping of the fusor, by
   * recursing into each argument
   * \param args The arguments to be updated
   * \return The updated arguments
   */
  Array<Expr> UpdateArgs(const Array<Expr>& args) {
    Array<Expr> new_args;
    new_args.reserve(args.size());
    for (const Expr& arg : args) {
      new_args.push_back(VisitExpr(arg));
    }
    return new_args;
  }

  /*!
   * \brief Add the input global variable and function to the new function list of the fusor, and
   * meanwhile resolve the name deduplication. We also discard the input function if some function
   * with the same name as the input function structurally equals to it.
   * \param gv The global variable corresponding to the new function to be added
   * \param func The new function to be added
   * \return The pair of the new global variable and the new function. Or a previously added pair if
   * the input function structurally equals to the previously added function.
   */
  std::pair<GlobalVar, Function> AddFuncWithDeduplication(GlobalVar gv, Function func) {
    std::string name = gv->name_hint;
    int suffix = 0;

    while (true) {
      auto it = new_functions_.find(name);
      if (it == new_functions_.end()) {
        if (gv->name_hint != name) {
          gv = GlobalVar(name);
          FunctionNode* p_func = func.CopyOnWrite();
          p_func->name = gv;
        }
        return std::make_pair(std::move(gv), std::move(func));
      }

      Function existing_func = (*it).second.second;
      if (structural_equal_(func, existing_func)) {
        LOG(INFO) << "[2]. deduplicate function " << gv->name_hint;
        return std::make_pair((*it).second.first, (*it).second.second);
      }

      std::ostringstream os;
      os << gv->name_hint << "_" << ++suffix;
      name = os.str();
    }
  }

 private:
  /*! \brief The IRModule. */
  IRModule mod_;
  /*! \brief Internal arena. */
  support::Arena arena_;
  /*! \brief The group assignment map. */
  std::unordered_map<const Object*, GraphPartitioner::Group*> obj2group_;
  /*! \brief Internal function information map. */
  std::unordered_map<GraphPartitioner::Group*, FunctionCreator> group2func_;
  /*! \brief The new global variables and functions to be added to the module */
  std::unordered_map<std::string, std::pair<GlobalVar, Function>> new_functions_;
  /*! \brief The structural equality checker */
  StructuralEqual structural_equal_;
};

IRModule FuseOps(IRModule mod, int opt_level, size_t max_fuse_depth) {
  support::Arena arena;

  // Step 1. Create the indexed-forward graph according to the input IRModule.
  IndexedForwardGraph graph = GraphCreator::Create(mod, &arena);

  // Step 2. Partition the graph by applying the fusion algorithm.
  std::vector<GraphPartitioner::Group*> groups =
      GraphPartitioner(&arena, opt_level, max_fuse_depth).Partition(graph);

  LOG(INFO) << "number of groups: " << groups.size();
  for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
    if (groups[i]->FindRoot() == groups[i]) {
      LOG(INFO) << "group[" << i << "] has " << groups[i]->num_nodes;
    }
  }

  // Step 3. Transform the IRModule by fusing the operators in accordance with the graph partition
  // results.
  mod = OperatorFusor(mod, graph, groups).Transform();

  const auto* f = runtime::Registry::Get("script.AsRelaxScript");
  String s = (*f)(mod, false);
  LOG(INFO) << "After FuseOps:\n" << s;

  // TODO(ruihang): unit tests: 1. name duplication

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

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
  static IndexedForwardGraph Create(support::Arena* arena, const IRModule& mod) {
    GlobalVar main_global_var = mod->GetGlobalVar("main");
    Function body = Downcast<Function>(mod->Lookup(main_global_var));
    IndexedForwardGraphCreator creator(arena, mod);
    creator.VisitExpr(body);
    // creator.graph_.DebugDump();
    return creator.graph_;
  }

 private:
  void VisitExpr_(const ConstantNode* op) final {
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

  void VisitExpr_(const SeqExprNode* op) {
    for (BindingBlock block : op->blocks) {
      this->VisitBindingBlock(block);
    }
    // Don't need to visit return value
    // this->VisitExpr(op->body);
  }

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
      const Expr& shape = op->args[0];
      GlobalVar global_var = Downcast<GlobalVar>(op->args[1]);
      tir::PrimFunc func = Downcast<tir::PrimFunc>(original_mod_->Lookup(global_var));
      const Tuple& args = Downcast<Tuple>(op->args[2]);
      int func_pattern = func->GetAttr<Integer>("op_pattern").value_or(OpPatternKind::kOpaque);

      // TODO(siyuan): Check the integer data is valid
      op_pattern = static_cast<OpPatternKind>(func_pattern);

      for (const Expr& arg : args->fields) {
        this->UpdateEdge(arg, node, op_pattern);
        this->VisitExpr(arg);
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
    LOG(INFO) << "Dataflow block expects no Control flow inside.";
  }

 private:
  explicit IndexedForwardGraphCreator(support::Arena* arena, const IRModule& mod)
      : original_mod_(mod), arena_(arena) {}

  // Helper functions to maintain IndexedForwardGraph
  /*!
   * \brief The
   * \param node The Relax IR nodes
   * \param parent The parent node in the IndexedForwardGraph.
   *               The source is external if the parent is nullptr.
   * \param pattern The relation pattern between the node and its parent.
   */
  void UpdateEdge(Expr node, IndexedForwardGraph::Node* parent, OpPatternKind pattern) {
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
  const IRModule& original_mod_;
  /*! \brief Allocator of all the internal node object */
  support::Arena* arena_;
  /*! \brief The output graph */
  IndexedForwardGraph graph_;
  /*! \brief current binding var */
  Optional<Var> cur_binding_var_ = NullOpt;
};

class FuseMutator : public ExprMutator {
 public:
  // Run the transform
  static IRModule Transform(const IRModule& mod, int fuse_opt_level, size_t max_fuse_depth) {
    // setup the group map.
    FuseMutator fuse_mutator(mod);
    GlobalVar main_global_var = mod->GetGlobalVar("main");
    Function main_func = Downcast<Function>(mod->Lookup(main_global_var));
    auto graph = IndexedForwardGraphCreator::Create(&fuse_mutator.arena_, mod);
    auto groups =
        GraphPartitioner(&fuse_mutator.arena_, fuse_opt_level, max_fuse_depth).Partition(graph);
    for (size_t nid = 0; nid < graph.post_dfs_order.size(); ++nid) {
      ICHECK(graph.post_dfs_order[nid]->ref != nullptr);
      fuse_mutator.gmap_[graph.post_dfs_order[nid]->ref] = groups[nid];
    }
    // The following line can be used for debug.
    GroupDebugDumper::Dump(main_func, fuse_mutator.gmap_);

    return mod;
  }

 private:
  explicit FuseMutator(const IRModule& mod) : original_mod_(mod) {}

 private:
  // Debug function, dump the group assignment in text.
  class GroupDebugDumper : public ExprVisitor {
   public:
    static void Dump(const Function& func,
                     const std::unordered_map<const Object*, GraphPartitioner::Group*>& gmap) {
      GroupDebugDumper dumper(gmap);
      for (const Var& pram : func->params) {
        // Skip function params since they are always a single group
        dumper.skip_objects_.insert(pram.get());
      }
      dumper(func);
      LOG(INFO) << "Group partition results:\n" << dumper.os_.str();
    }

   private:
    explicit GroupDebugDumper(
        const std::unordered_map<const Object*, GraphPartitioner::Group*>& gmap)
        : gmap_(gmap) {}

    void TryPrintGroup(const ObjectRef& object) {
      if (skip_objects_.count(object.get())) return;
      auto it = gmap_.find(object.get());
      if (it == gmap_.end()) return;
      const GraphPartitioner::Group* group = it->second->FindRoot();
      if (const auto* var = object.as<VarNode>()) {
        os_ << var->name_hint();
      } else {
        os_ << object;
      }
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

  /*! \brief Internal arena. */
  support::Arena arena_;
  /*! \brief The group assignment map. */
  std::unordered_map<const Object*, GraphPartitioner::Group*> gmap_;
  /*! \brief The IRModule. */
  IRModule original_mod_;
};

namespace transform {

Pass FuseOps(int fuse_opt_level) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) {
        int opt_level = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
        auto max_fuse_depth = pc->GetConfig("relax.FuseOps.max_depth", Integer(kMaxFusedOps));
        return FuseMutator::Transform(m, opt_level, max_fuse_depth.value());
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

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
#include "../utils.h"

namespace tvm {
namespace tir {

class IndexMapCollector : public StmtExprVisitor {
 public:
  IndexMapCollector(const Buffer& buffer, arith::Analyzer* analyzer)
      : buffer_(buffer), analyzer_(analyzer) {}

  Optional<BlockRV> GetBlockRV(const Schedule& sch, String global_var_name) {
    if (index_map_.defined()) {
      return sch->GetBlock(tgt_realize_->block->name_hint, global_var_name);
    }
    return NullOpt;
  }

  int GetBufferIndex() {
    if (index_map_.defined()) {
      for (int i = 0; i < static_cast<int>(tgt_realize_->block->reads.size()); i++) {
        if (tgt_realize_->block->reads[i]->buffer.same_as(buffer_)) {
          return i;
        }
      }
    }
    return -1;
  }

  Optional<IndexMap> GetIndexMap() { return index_map_; }

 private:
  void VisitStmt_(const ForNode* op) final {
    loops_.push_back(GetRef<For>(op));
    StmtVisitor::VisitStmt_(op);
    loops_.pop_back();
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize outer_block = GetRef<BlockRealize>(op);
    std::swap(outer_block, cur_realize_);
    StmtVisitor::VisitStmt_(op);
    std::swap(cur_realize_, outer_block);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (op->buffer.same_as(buffer_)) {
      Map<Var, PrimExpr> subst_map;
      for (int i = 0; i < static_cast<int>(cur_realize_->iter_values.size()); i++) {
        subst_map.Set(cur_realize_->block->iter_vars[i]->var, cur_realize_->iter_values[i]);
      }
      Array<PrimExpr> subst_indices;
      for (const PrimExpr& e : op->indices) {
        subst_indices.push_back(Substitute(e, subst_map));
      }
      index_map_ = SuggestIndexMap(buffer_, subst_indices, loops_, cur_realize_->predicate, analyzer_);
      tgt_realize_ = cur_realize_;
    }
  }

  Buffer buffer_;
  arith::Analyzer* analyzer_;
  Array<For> loops_;
  BlockRealize cur_realize_;
  BlockRealize tgt_realize_;
  Optional<IndexMap> index_map_;
};

bool LayoutRewrite(const Schedule& sch) {
  std::vector<std::pair<StmtSRef, String>> results;
  for (const auto& kv : sch->mod()->functions) {
    GlobalVar g_var = kv.first;
    BaseFunc base_func = kv.second;
    if (const auto* f = base_func.as<PrimFuncNode>()) {
      auto layout_free_buffer_index = Downcast<Array<Integer>>(
          f->GetAttr("layout_rewrite_buffers", Array<Integer>()));
      if (!layout_free_buffer_index.defined()) {
        continue;
      }
      Array<Buffer> layout_free_buffers;
      for (const auto& i: layout_free_buffer_index) {
        layout_free_buffers.push_back(f->buffer_map.Get(f->params[i->value]).value());
      }
      String func_name = g_var->name_hint;
      arith::Analyzer analyzer;
      for (const Buffer& buffer : layout_free_buffers) {
        IndexMapCollector collector(buffer, &analyzer);
        collector(f->body);
        Optional<IndexMap> index_map_opt = collector.GetIndexMap();
        if (index_map_opt.defined()) {
          BlockRV block_rv = collector.GetBlockRV(sch, func_name).value();
          int buffer_index = collector.GetBufferIndex();
          sch->TransformLayoutWithPreProc(block_rv, buffer_index, BufferIndexType::kRead,
                               index_map_opt.value());
        }
      }
    }
  }
  return true;
}

}  // namespace tir

namespace meta_schedule {

/*! \brief Layout Rewrite. */
class LayoutRewriteNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final { return tir::LayoutRewrite(sch); }

  static constexpr const char* _type_key = "meta_schedule.LayoutRewrite";
  TVM_DECLARE_FINAL_OBJECT_INFO(LayoutRewriteNode, PostprocNode);
};

Postproc Postproc::LayoutRewrite() {
  ObjectPtr<LayoutRewriteNode> n = make_object<LayoutRewriteNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(LayoutRewriteNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocLayoutRewrite").set_body_typed(Postproc::LayoutRewrite);

}  // namespace meta_schedule
}  // namespace tvm

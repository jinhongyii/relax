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

std::pair<Stmt, Block> GeneratePreProcBody(const Buffer& old_buffer, const Buffer& new_buffer,
                                           const IndexMap& index_map) {
  ICHECK_EQ(index_map->initial_indices.size(), old_buffer->shape.size());
  Array<PrimExpr> src_indices;
  for (tir::Var v : index_map->initial_indices) {
    src_indices.push_back(v);
  }
  tir::Stmt body = tir::BufferStore(new_buffer, tir::BufferLoad(old_buffer, src_indices),
                                    index_map->final_indices);
  Array<tir::IterVar> block_iters;
  int n = old_buffer->shape.size();
  for (int i = 0; i < n; i++) {
    block_iters.push_back(tir::IterVar(Range::FromMinExtent(0, old_buffer->shape[i]),
                                       index_map->initial_indices[i], tir::kDataPar));
  }
  Array<Range> read_region;
  for (int i = 0; i < n; i++) {
    read_region.push_back(Range::FromMinExtent(0, old_buffer->shape[i]));
  }
  Array<Range> write_region = index_map->MapRanges(read_region);
  body = Block(block_iters, {BufferRegion(old_buffer, read_region)},
               {BufferRegion(new_buffer, write_region)}, "layout_rewrite", body, NullOpt, {}, {},
               {{"preproc", Bool(true)}});
  auto infered_access_regions = GetBlockReadWriteRegion(
      Downcast<Block>(body), {{new_buffer->data, new_buffer}, {old_buffer->data, old_buffer}});
  ObjectPtr<BlockNode> preproc_block = make_object<BlockNode>(*body.as<BlockNode>());
  preproc_block->reads = infered_access_regions[0];
  preproc_block->writes = infered_access_regions[1];
  body = Block(preproc_block);
  Array<PrimExpr> loop_vars;
  for (int i = 0; i < n; i++) {
    loop_vars.push_back(index_map->initial_indices[i].copy_with_suffix("_o"));
  }
  body = tir::BlockRealize(loop_vars, Bool(true), Downcast<tir::Block>(body));
  for (int i = n - 1; i >= 0; i--) {
    body = For(Downcast<Var>(loop_vars[i]), 0, old_buffer->shape[i], tir::ForKind::kSerial, body);
  }
  return std::make_pair(body, Block(preproc_block));
}

class TransformLayoutRewriter : private StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the access to the buffer after the transformation
   * \param scope_stmt The parent statement that contains all accesses to the target buffer
   * \param old_buffer The target buffer before transformation
   * \param new_buffer The new buffer after transformation
   * \param index_map The transformation applied to the buffer
   * \return The new AST rooting at the original parent scope and the map from the old block to the
   * new block
   */
  static std::pair<Stmt, Map<Block, Block>> Rewrite(const Stmt& scope_stmt,
                                                    const Buffer& old_buffer,
                                                    const Buffer& new_buffer,
                                                    const IndexMap& index_map,
                                                    Optional<Stmt> preproc_body = NullOpt) {
    TransformLayoutRewriter rewriter(old_buffer, new_buffer, index_map, preproc_body);
    Stmt result = rewriter(scope_stmt);
    return {result, rewriter.block_sref_reuse_};
  }

 private:
  TransformLayoutRewriter(const Buffer& old_buffer, const Buffer& new_buffer,
                          const IndexMap& index_map, Optional<Stmt> preproc_body = NullOpt)
      : old_buffer_(old_buffer),
        new_buffer_(new_buffer),
        index_map_(index_map),
        preproc_body_(preproc_body),
        buffer_data_to_buffer_{{new_buffer->data, new_buffer}} {}

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices) {
    *buffer = new_buffer_;
    *indices = index_map_->MapIndices(*indices);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad buffer_load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (buffer_load->buffer.same_as(old_buffer_)) {
      auto* n = buffer_load.CopyOnWrite();
      RewriteBufferAccess(&n->buffer, &n->indices);
    }
    return std::move(buffer_load);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore buffer_store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (buffer_store->buffer.same_as(old_buffer_)) {
      auto* n = buffer_store.CopyOnWrite();
      RewriteBufferAccess(&n->buffer, &n->indices);
    }
    return std::move(buffer_store);
  }

  void RewriteAccessRegion(Array<BufferRegion>* old_access_regions,
                           const Array<BufferRegion>& infered_access_regions) {
    auto fmutate = [this, &infered_access_regions](const BufferRegion& buffer_region) {
      if (buffer_region->buffer.same_as(old_buffer_)) {
        ICHECK(infered_access_regions.size() == 1);
        return infered_access_regions[0];
      }
      return buffer_region;
    };
    (*old_access_regions).MutateByApply(fmutate);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    auto infered_access_regions = GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    auto* n = block.CopyOnWrite();
    RewriteAccessRegion(&n->reads, infered_access_regions[0]);
    RewriteAccessRegion(&n->writes, infered_access_regions[1]);
    if (n->name_hint == "root" && preproc_body_.defined()) {
      n->body = SeqStmt::Flatten(preproc_body_.value(), n->body);
      n->alloc_buffers.push_back(new_buffer_);
    }
    block_sref_reuse_.Set(GetRef<Block>(op), block);
    return std::move(block);
  }

  const Buffer& old_buffer_;
  const Buffer& new_buffer_;
  const IndexMap& index_map_;
  Optional<Stmt> preproc_body_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Block, Block> block_sref_reuse_;
};

class BufferIsSubregionError : public ScheduleError {
 public:
  explicit BufferIsSubregionError(IRModule mod, Buffer buffer) : mod_(mod), buffer_(buffer) {}

  String FastErrorString() const final {
    return "ScheduleError: The input buffer is defined in `match_buffer` of a block, it is expected"
           " to be a function parameter or allocated by a block";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: The input buffer " << buffer_->name << " is defined in `match_buffer` of "
       << "a block, it is expected to be a function parameter or allocated by a block.";
    return os.str();
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  Buffer buffer_;
};

class BufferNotParamError : public ScheduleError {
 public:
  explicit BufferNotParamError(IRModule mod, Buffer buffer) : mod_(mod), buffer_(buffer) {}

  String FastErrorString() const final {
    return "ScheduleError: The input buffer is allocated by a block. It is expected"
           " to be a function parameter.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: The input buffer " << buffer_->name
       << " is allocated by a block. it is"
          " expected to be a function parameter.";
    return os.str();
  }

  Array<ObjectRef> LocationsOfInterest() const final { return {}; }
  IRModule mod() const final { return mod_; }

 private:
  IRModule mod_;
  Buffer buffer_;
};

void TransformLayout(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                     BufferIndexType buffer_index_type, const IndexMap& index_map) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_ptr, block_sref);
  Buffer old_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block_ptr), buffer_index,
                         buffer_index_type == BufferIndexType::kRead ? false : true);
  Optional<StmtSRef> defining_site_sref;
  bool is_alloc;
  std::tie(defining_site_sref, is_alloc) = GetBufferDefiningSite(block_sref, old_buffer);
  if (defining_site_sref.defined() && !is_alloc) {
    throw BufferIsSubregionError(self->mod, old_buffer);
  }

  StmtSRef scope_sref = defining_site_sref.defined()
                            ? defining_site_sref.value()
                            : GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);

  // Step 1: Infer the shape of the new buffer
  ObjectPtr<BufferNode> new_buffer_node = make_object<BufferNode>(*(old_buffer.get()));
  new_buffer_node->shape = index_map->MapShape(old_buffer->shape);
  Buffer new_buffer{new_buffer_node};

  // Step 2: Rewrite access indices and regions of the buffer
  Stmt new_stmt;
  Map<Block, Block> block_sref_reuse;
  std::tie(new_stmt, block_sref_reuse) = TransformLayoutRewriter::Rewrite(
      GetRef<Block>(scope_block), old_buffer, new_buffer, index_map);
  Block new_scope_block = Downcast<Block>(new_stmt);

  // Step 3: Rewrite alloc_buffer of the block or buffer_map of the PrimFunc.
  if (defining_site_sref.defined()) {
    auto* n = new_scope_block.CopyOnWrite();
    n->alloc_buffers.MutateByApply([&old_buffer, &new_buffer](const Buffer& buffer) {
      if (buffer.same_as(old_buffer)) {
        return new_buffer;
      }
      return buffer;
    });
    block_sref_reuse.Set(GetRef<Block>(scope_block), new_scope_block);
  } else {
    GlobalVar g_var;
    GetRootPrimFunc(self->mod, scope_block, &g_var);
    IRModuleNode* new_mod = self->mod.CopyOnWrite();
    MapNode* new_map = new_mod->functions.CopyOnWrite();
    PrimFunc ref_new_func = Downcast<PrimFunc>(std::move(new_map->at(g_var)));
    PrimFuncNode* new_func = ref_new_func.CopyOnWrite();
    MapNode* new_buffer_map = new_func->buffer_map.CopyOnWrite();
    for (auto it = new_buffer_map->begin(); it != new_buffer_map->end(); ++it) {
      if ((*it).second.same_as(old_buffer)) {
        (*it).second = new_buffer;
      }
    }
    new_map->at(g_var) = std::move(ref_new_func);
  }

  // Step 4: Replace the scope block with the new block
  self->Replace(scope_sref, new_scope_block, block_sref_reuse);
}

void TransformLayoutWithPreProc(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                                BufferIndexType buffer_index_type, const IndexMap& index_map) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_ptr, block_sref);
  Buffer old_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block_ptr), buffer_index,
                         buffer_index_type == BufferIndexType::kRead ? false : true);
  Optional<StmtSRef> defining_site_sref;
  bool is_alloc;
  std::tie(defining_site_sref, is_alloc) = GetBufferDefiningSite(block_sref, old_buffer);
  if (is_alloc) {
    throw BufferNotParamError(self->mod, old_buffer);
  }

  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);

  // Step 1: Infer the shape of the new buffer
  ObjectPtr<BufferNode> new_buffer_node = make_object<BufferNode>(*(old_buffer.get()));
  ObjectPtr<VarNode> new_var = make_object<VarNode>(*old_buffer->data.get());
  new_buffer_node->shape = index_map->MapShape(old_buffer->shape);
  new_buffer_node->data = Var(new_var);
  Buffer new_buffer{new_buffer_node};
  // Step 2: Generate Preproc body
  Block preproc_block;
  Stmt preproc_body;
  std::tie(preproc_body, preproc_block) = GeneratePreProcBody(old_buffer, new_buffer, index_map);
  // Step 3: Rewrite access indices and regions of the buffer
  Stmt new_stmt;
  Map<Block, Block> block_sref_reuse;
  std::tie(new_stmt, block_sref_reuse) = TransformLayoutRewriter::Rewrite(
      GetRef<Block>(scope_block), old_buffer, new_buffer, index_map, preproc_body);
  Block new_scope_block = Downcast<Block>(new_stmt);
  // Step 4: Replace the scope block with the new block
  self->Replace(scope_sref, new_scope_block, block_sref_reuse);
  StmtSRef preproc_block_sref = self->stmt2ref.at(preproc_block.get());
  BlockInfo& block_info = self->block_info[preproc_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, preproc_block_sref);
  block_info.region_cover = true;
  block_info.scope->stage_pipeline = true;
}

/******** InstructionKind Registration ********/

struct TransformLayoutTraits : public UnpackedInstTraits<TransformLayoutTraits> {
  static constexpr const char* kName = "TransformLayout";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 3;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      Integer buffer_index_type, IndexMap index_map) {
    return sch->TransformLayout(block_rv, buffer_index,
                                static_cast<BufferIndexType>(buffer_index_type->value), index_map);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer buffer_index_type, IndexMap index_map) {
    PythonAPICall py("transform_layout");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index);
    py.Input("buffer_index_type",
             BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value)));
    py.Input("index_map", index_map->ToPythonString());
    return py.Str();
  }

 public:
  static ObjectRef AttrsAsJSON(const Array<ObjectRef>& attrs) {
    Array<ObjectRef> attrs_record;
    attrs_record.reserve(kNumAttrs);
    attrs_record.push_back(attrs[0]);
    attrs_record.push_back(attrs[1]);
    attrs_record.push_back(String(::tvm::SaveJSON(attrs[2])));
    return std::move(attrs_record);
  }

  static Array<ObjectRef> AttrsFromJSON(const ObjectRef& attrs_record_) {
    Array<ObjectRef> attrs_record = Downcast<Array<ObjectRef>>(attrs_record_);
    Array<ObjectRef> attrs;
    attrs.push_back(attrs_record[0]);
    attrs.push_back(attrs_record[1]);
    attrs.push_back(::tvm::LoadJSON(Downcast<String>(attrs_record[2])));
    return attrs;
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct TransformLayoutWithPreProcTraits
    : public UnpackedInstTraits<TransformLayoutWithPreProcTraits> {
  static constexpr const char* kName = "TransformLayoutWithPreProc";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 3;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, Integer buffer_index,
                                      Integer buffer_index_type, IndexMap index_map) {
    return sch->TransformLayoutWithPreProc(
        block_rv, buffer_index, static_cast<BufferIndexType>(buffer_index_type->value), index_map);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer buffer_index_type, IndexMap index_map) {
    PythonAPICall py("transform_layout_with_preproc");
    py.Input("block", block_rv);
    py.Input("buffer_index", buffer_index);
    py.Input("buffer_index_type",
             BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value)));
    py.Input("index_map", index_map->ToPythonString());
    return py.Str();
  }

 public:
  static ObjectRef AttrsAsJSON(const Array<ObjectRef>& attrs) {
    Array<ObjectRef> attrs_record;
    attrs_record.reserve(kNumAttrs);
    attrs_record.push_back(attrs[0]);
    attrs_record.push_back(attrs[1]);
    attrs_record.push_back(String(::tvm::SaveJSON(attrs[2])));
    return std::move(attrs_record);
  }

  static Array<ObjectRef> AttrsFromJSON(const ObjectRef& attrs_record_) {
    Array<ObjectRef> attrs_record = Downcast<Array<ObjectRef>>(attrs_record_);
    Array<ObjectRef> attrs;
    attrs.push_back(attrs_record[0]);
    attrs.push_back(attrs_record[1]);
    attrs.push_back(::tvm::LoadJSON(Downcast<String>(attrs_record[2])));
    return attrs;
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(TransformLayoutTraits);
TVM_REGISTER_INST_KIND_TRAITS(TransformLayoutWithPreProcTraits);

}  // namespace tir
}  // namespace tvm

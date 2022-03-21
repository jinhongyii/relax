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
 * \file src/tir/transforms/fuse_primfuncs.cc
 * \brief fuse multiple function in an IRModule
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../ir/functor_common.h"

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
  std::unordered_map<const VarNode*, Buffer> buffer_var_map_;

  static Array<BufferRegion> UnionAccessRegion(Array<BufferRegion> regions) {
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
 * \brief Fuse multiple PrimFuncs into fused one
 * \param funcs The PrimFuncs should be inlined. The list guides the order of fusion.
 * \param param_map A map indicate how data exchange between functions.
 *                  The map is from consumer params to the producer params.
 * \return The fused function.
 */
PrimFunc FusePrimFuncs(const Array<PrimFunc>& funcs, const Map<Var, Var>& param_map) {
  ICHECK(!funcs.empty());
  if (funcs.size() == 1) {
    ICHECK(param_map.empty());
    return funcs[0];
  }

  // Step 1. Checking functions are all schedulable funcs. i.e. the body of func is root block
  for (const PrimFunc& func : funcs) {
    CHECK(func->body->IsInstance<BlockRealizeNode>())
        << "Only schedulable functions (whose body is the root block) can be fused";
  }

  // Step 2. Create a map from param var to the PrimFunc.
  //         Checking functions contain only buffer params
  Map<Var, PrimFunc> param2func;
  for (const PrimFunc& func : funcs) {
    for (const Var& param : func->params) {
      CHECK(param.dtype().is_handle()) << "Functions can only contain buffer params.";
      param2func.Set(param, func);
    }
  }

  // Step 3. Checking the buffer are the same between param_map
  auto param2buffer = [&param2func](const Var& param) -> Buffer {
    auto it = param2func.find(param);
    ICHECK(it != param2func.end());
    const PrimFunc& func = (*it).second;
    auto it2 = func->buffer_map.find(param);
    ICHECK(it2 != func->buffer_map.end());
    return (*it2).second;
  };
  auto buffer_equal = [](const Buffer& lhs, const Buffer& rhs) -> bool {
    return StructuralEqual()(lhs, rhs);
  };
  for (const auto& kv : param_map) {
    const Var& producer_param = kv.second;
    const Var& consumer_param = kv.first;
    const Buffer& producer_buffer = param2buffer(producer_param);
    const Buffer& consumer_buffer = param2buffer(consumer_param);
    CHECK(buffer_equal(consumer_buffer, producer_buffer))
        << "Buffer attrs are ont equal between " << consumer_buffer << " " << producer_buffer
        << ".";
  }

  // Step 4. Creating function dependence
  std::unordered_map<const PrimFuncNode*, int> func_dep_count;
  std::unordered_map<const PrimFuncNode*, Array<PrimFunc>> func_dep_edges;
  for (const PrimFunc& func : funcs) {
    func_dep_count[func.get()] = 0;
    func_dep_edges[func.get()] = Array<PrimFunc>();
  }
  for (const auto& kv : param_map) {
    const Var& producer_param = kv.second;
    const Var& consumer_param = kv.first;
    // Have checked before so don't need to check here
    const PrimFunc& producer_func = param2func.at(producer_param);
    const PrimFunc& consumer_func = param2func.at(consumer_param);
    func_dep_count[consumer_func.get()] += 1;
    func_dep_edges[producer_func.get()].push_back(consumer_func);
  }

  // Step 5. Collect all param and coressponding buffer which should be inlined.
  std::unordered_set<const VarNode*> inlined_params;
  for (const auto& kv : param_map) {
    const Var& producer_param = kv.second;
    const Var& consumer_param = kv.first;
    inlined_params.insert(consumer_param.get());
    inlined_params.insert(producer_param.get());
  }

  // Step 6. Generate fused function
  Array<Stmt> bodies;
  Array<Var> params;
  Map<Var, Buffer> buffer_map;
  Array<Buffer> alloc_buffers;

  for (const PrimFunc& func : funcs) {
    // Step 6.1. Checking dependency
    CHECK_EQ(func_dep_count.at(func.get()), 0)
        << "The dependecy of function is not satisfied. "
           "Please change the param mapping or the order of input functions.\n"
           "The function is \n "
        << func;

    BlockRealize root_realize = Downcast<BlockRealize>(func->body);
    Block root_block = root_realize->block;

    // Step 6.2. Insert alloc_buffers in original func
    alloc_buffers.insert(alloc_buffers.end(),                //
                         root_block->alloc_buffers.begin(),  //
                         root_block->alloc_buffers.end());

    // Step 6.3. Update param and buffer map
    for (const Var& param : func->params) {
      const Buffer& buffer = func->buffer_map.at(param);
      if (inlined_params.count(param.get())) {
        // If the param should be inlined, then alloc the coressponding buffer.
        // Only alloc once for the same producer and consumer param.
        if (param_map.find(param) == param_map.end()) {
          alloc_buffers.push_back(buffer);
        }
      } else {
        // Else add the param to the final param list and update buffer map
        params.push_back(param);
        buffer_map.Set(param, buffer);
      }
    }

    // Step 6.4. Don't change the bodies but push back into array
    bodies.push_back(root_block->body);

    // Step 6.5. Updating left dependency
    for (const PrimFunc& consumer_func : func_dep_edges.at(func.get())) {
      func_dep_count.at(consumer_func.get()) -= 1;
    }
  }

  // Step 7. Creating new body
  Stmt body = Block({}, {}, {}, "root", SeqStmt::Flatten(bodies), NullOpt, alloc_buffers);
  body = BlockRealize({}, Bool(true), Downcast<Block>(body));

  // Step 8. Substitute buffers
  Map<Buffer, Buffer> bmap;
  for (const auto& kv : param_map) {
    const Var& producer_param = kv.second;
    const Var& consumer_param = kv.first;
    const Buffer& producer_buffer = param2buffer(producer_param);
    const Buffer& consumer_buffer = param2buffer(consumer_param);
    bmap.Set(consumer_buffer, producer_buffer);
  }
  body = BufferSubstituter::Substitute(bmap, body);

  // Step 9. Update func_attr
  // Only consider global_symbol and noalias for now.
  std::string global_name = "fused";
  for (const PrimFunc func : funcs) {
    // global_symbol
    Optional<String> name = func->attrs.GetAttr<String>("global_symbol");
    if (name.defined()) global_name += "_" + name.value();
  }
  Map<String, ObjectRef> attr_map;
  attr_map.Set("tir.noalias", const_true());
  if (global_name != "fused") {
    attr_map.Set("global_symbol", String(global_name));
  }

  return PrimFunc(params, body, VoidType(), buffer_map, DictAttrs(attr_map));
}

/**************** FFI ****************/
TVM_REGISTER_GLOBAL("tir.FusePrimFuncs").set_body_typed(FusePrimFuncs);
}  // namespace tir
}  // namespace tvm

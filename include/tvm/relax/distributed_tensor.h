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
 * \file tvm/relax/distributed.h
 * \brief Data structures for constructing distributed tensor in relax programs.
 */

#ifndef TVM_RELAX_DISTRIBUTED_H_
#define TVM_RELAX_DISTRIBUTED_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>

namespace tvm {
namespace relax {

/*
 * \brief Device mesh express a view of topology of devices, represented by an n-d matrix of
 * device ids
 */
class DeviceMeshNode : public GlobalInfoNode {
 public:
  /*! \brief logical shape of the mesh*/
  ShapeTuple shape;
  
  /*! \brief device ids in the mesh*/
  Array<Integer> device_ids;

  /*! \brief Optionally use range(start, end, step) to represent device_ids*/
  Optional<Integer> device_start;
  Optional<Integer> device_end;
  Optional<Integer> device_step;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("shape", &shape);
    v->Visit("device_ids", &device_ids);
    v->Visit("device_start", &device_start);
    v->Visit("device_end", &device_end);
    v->Visit("device_step", &device_step);
  }
  static constexpr const char* _type_key = "relax.distributed.DeviceMesh";

  bool SEqualReduce(const DeviceMeshNode* other, SEqualReducer equal) const {
    if (shape.size() != other->shape.size()) {
      return false;
    }
    for (int i = 0; i < shape.size(); i++) {
      if (!equal(shape[i], other->shape[i])) {
        return false;
      }
    }
    return equal(device_ids, other->device_ids);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(device_ids);
    for (int i = 0; i < shape.size(); i++) {
      hash_reduce(shape[i]);
    }
  }

  TVM_DECLARE_FINAL_OBJECT_INFO(DeviceMeshNode, GlobalInfoNode);
};

/*!
 * \brief Managed reference to a DeviceMesh.
 * \sa DeviceMeshNode
 */
class DeviceMesh : public GlobalInfo {
 public:
  TVM_DLL DeviceMesh(ShapeTuple shape, Array<Integer> device_ids);
  TVM_DLL DeviceMesh(ShapeTuple shape, Integer device_start, Integer device_end, Integer device_step);
  TVM_DEFINE_OBJECT_REF_METHODS(DeviceMesh, GlobalInfo, DeviceMeshNode);
};

/*! \brief Describes how data is distributed in each dimension of the device mesh*/
class PlacementNode : public Object {
 public:
  /*! \brief placement for each dim of device mesh. -1 represents replica, and integer >=0 represents sharding dimension on tensor*/
  Array<Integer> dim_placement;
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("dim_placement", &dim_placement); }

  bool SEqualReduce(const PlacementNode* other, SEqualReducer equal) const {
    return equal(dim_placement, other->dim_placement);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(dim_placement); }

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "relax.distributed.Placement";
  TVM_DECLARE_FINAL_OBJECT_INFO(PlacementNode, Object);
};

/*!
 * \brief Managed reference to a Placement.
 * \sa PlacementNode
 */
class Placement : public ObjectRef {
 public:
  TVM_DLL explicit Placement(Array<Integer> dim_placement);
  /*! \brief replica dim is printed as "R" and sharding dim is printed as "S[i]". So a text "S[1]R" can be translated into placement[1, -1]*/
  TVM_DLL explicit Placement(String text_format);
  TVM_DEFINE_OBJECT_REF_METHODS(Placement, ObjectRef, PlacementNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_H_
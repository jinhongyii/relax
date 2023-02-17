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
 * \brief Data structures for distributed relax programs.
 */
 
#ifndef TVM_RELAX_DISTRIBUTED_H_
#define TVM_RELAX_DISTRIBUTED_H_

#include <tvm/ir/module.h>
#include "tvm/ir/expr.h"

namespace tvm {
namespace relax {

/*
* \brief Device mesh express a view of topology of devices, usually represented by an n-d matrix of device ids
*/
class DeviceMeshNode: public GlobalInfoNode{
    public:
    /*! \brief device ids in the mesh*/
    Array<Integer> device_ids;
    /*! \brief logical shape of the mesh*/
    ShapeTuple shape;


    void VisitAttrs(tvm::AttrVisitor* v) { 
        v->Visit("shape", &shape);
        v->Visit("device_ids", &device_ids); 
    }
    static constexpr const char* _type_key = "DeviceMesh";
    TVM_DECLARE_FINAL_OBJECT_INFO(DeviceMeshNode, GlobalInfoNode);
};



class DeviceMesh: public GlobalInfo{
    public:
    DeviceMesh() = default;
    explicit DeviceMesh(Array<Expr> shape) : shape_(shape) {}
    TVM_DEFINE_OBJECT_REF_METHODS(DeviceMesh, GlobalInfo, DeviceMeshNode);
    };
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_H_
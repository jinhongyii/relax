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

#include <tvm/relax/distributed_tensor.h>

namespace tvm {
namespace relax {

DeviceMesh::DeviceMesh(ShapeTuple shape, Array<Integer> device_ids) {
  int prod = 1;
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    prod *= shape[i];
  }
  ObjectPtr<DeviceMeshNode> n = make_object<DeviceMeshNode>();
  CHECK(prod == device_ids.size())
      << "The number of device ids must match the product of the shape";
  n->shape = std::move(shape);
  n->device_ids = std::move(device_ids);
  data_ = std::move(n);
}

DeviceMesh::DeviceMesh(ShapeTuple shape, Integer device_start, Integer device_end,
                       Integer device_step) {
  CHECK(device_step->value > 0);
  ObjectPtr<DeviceMeshNode> n = make_object<DeviceMeshNode>();
  Array<Integer> device_ids;
  for (int i = device_start->value; i < device_end->value; i += device_step->value) {
    device_ids.push_back(i);
  }
  int prod = 1;
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    prod *= shape[i];
  }
  CHECK(prod == device_ids.size())
      << "The number of device ids must match the product of the shape";
  n->device_ids = std::move(device_ids);
  n->shape = std::move(shape);
  n->device_start = std::move(device_start);
  n->device_end = std::move(device_end);
  n->device_step = std::move(device_step);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DeviceMeshNode);
TVM_REGISTER_GLOBAL("relax.distributed.DeviceMesh")
    .set_body_typed([](ShapeTuple shape, Integer start, Integer end, Integer step) {
      return DeviceMesh(shape, start, end, step);
    });

Placement::Placement(Array<Integer> dim_placement) {
  ObjectPtr<PlacementNode> n = make_object<PlacementNode>();
  n->dim_placement = std::move(dim_placement);
  data_ = std::move(n);
}

Placement::Placement(String text_format) {
  Array<Integer> dim_placement;
  std::stringstream ss(text_format);
  while (true) {
    char indicator = 0;
    ss >> indicator;
    if (ss.eof()) {
      break;
    }
    if (indicator == 'R') {
      dim_placement.push_back(-1);
    } else if (indicator == 'S') {
      char lbracket;
      ss >> lbracket;
      CHECK(lbracket == '[');
      std::string substr;
      getline(ss, substr, ']');
      std::stringstream ss2(substr);
      int dim;
      ss2 >> dim;
      dim_placement.push_back(dim);
      CHECK(ss2.eof()) << "Invalid placement format";
    } else {
      LOG(FATAL) << "Invalid placement format";
    }
  }
  ObjectPtr<PlacementNode> n = make_object<PlacementNode>();
  n->dim_placement = std::move(dim_placement);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PlacementNode);
TVM_REGISTER_GLOBAL("relax.distributed.Placement").set_body_typed([](String text_format) {
  return Placement(text_format);
});
}  // namespace relax
}  // namespace tvm
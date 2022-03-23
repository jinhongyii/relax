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
* \file src/relax/transform/annotate_opkind.cc
* \brief Annotate OpKind for TIR functions
*/
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/analysis.h>

#include "tvm/tir/function.h"
#include "../../printer/text_printer.h"

namespace tvm {
namespace relax {


IRModule Annotate(IRModule m) {
  Map<GlobalVar, tir::PrimFunc> func_map;
  for (const std::pair<GlobalVar, BaseFunc>& pr : m->functions) {
    if (const auto* f = pr.second.as<tir::PrimFuncNode>()) {
      relay::OpPatternKind kind = AnalyzeOpPatternKind(GetRef<tir::PrimFunc>(f));
      tir::PrimFunc annotated_func = WithAttr(GetRef<tir::PrimFunc>(f), "op_pattern", Integer(static_cast<int>
    (kind)));
      func_map.Set(pr.first, annotated_func);
    }
  }
  IRModuleNode* n = m.CopyOnWrite();
  for (auto pr : func_map) {
    n->functions.Set(pr.first, pr.second);
  }
  return GetRef<IRModule>(n);
}

namespace transform {

Pass AnnotateOpKind() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return Annotate(mod); };
  return CreateModulePass(pass_func, 0, "VMShapeLower", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AnnotateOpKind").set_body_typed(AnnotateOpKind);

}  // namespace transform

}  // namespace relax
}  // namespace tvm

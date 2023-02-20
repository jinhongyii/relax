# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=redefined-builtin, invalid-name
"""Data structures for distributed tensor."""
import tvm
from typing import List, Union, Tuple
from tvm.ir.global_info import GlobalInfo
from tvm.runtime import ShapeTuple
from tvm.runtime.object import Object

from . import _ffi_api as ffi


@tvm._ffi.register_object("relax.distributed.DeviceMesh")
class DeviceMesh(GlobalInfo):
    """Device mesh express a view of topology of devices, represented by an n-d matrix of device ids.

    Parameters
    ----------
    shape: Union[ShapeTuple, List[int], Tuple[int]]
        Logical shape of device mesh
    device_start: int
    device_end: int
    device_step: int
        range(device_start, device_end, device_step) represents the device id in the mesh
    """

    def __init__(self, shape: Union[ShapeTuple, List[int], Tuple[int]], device_start: int, device_end: int, device_step: int = 1):
        if isinstance(shape, (list, tuple)):
            shape = ShapeTuple(shape)
        self.__init_handle_by_constructor__(
            ffi.DeviceMesh, shape, device_start, device_end, device_step  # type: ignore
        )


@tvm._ffi.register_object("relax.distributed.Placement")
class Placement(Object):
    """ Describes how data is distributed in each dimension of the device mesh

    Parameters
    ----------
    text_format: str
        The text format of placement.
    """
    def __init__(self, text_format: str):
        self.__init_handle_by_constructor__(
            ffi.Placement, text_format  # type: ignore
        )
    
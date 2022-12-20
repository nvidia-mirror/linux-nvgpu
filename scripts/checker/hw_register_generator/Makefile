#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

repo_path := $(abspath $(shell dirname $(lastword $(MAKEFILE_LIST)))/../../../../..)
kernel_path := $(repo_path)/kernel
checker_path := $(repo_path)/qnx/src/resmgrs/nvrm/nvgpu_rmos/libnvgpu/src/checker

flags := -I$(kernel_path)/nvgpu/drivers/gpu/nvgpu/include

srcs  := hw_register_generator.c

build:
	gcc $(srcs) $(flags) -o hw_registers_table_generator

generate:
	./hw_registers_table_generator "$(checker_path)/hw_registers_table.c"

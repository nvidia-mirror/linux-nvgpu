/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef NVGPU_GSP_PRIV
#define NVGPU_GSP_PRIV

#include <nvgpu/lock.h>

#define GSP_DEBUG_BUFFER_QUEUE	3U
#define GSP_DMESG_BUFFER_SIZE	0x1000U

struct gsp_fw {
	/* gsp ucode */
	struct nvgpu_firmware *code;
	struct nvgpu_firmware *data;
	struct nvgpu_firmware *manifest;
};

/* GSP descriptor's */
struct nvgpu_gsp {
	struct gk20a *g;

	struct gsp_fw gsp_ucode;
	struct nvgpu_falcon *gsp_flcn;

	bool isr_enabled;
	struct nvgpu_mutex isr_mutex;
};
#endif /* NVGPU_GSP_PRIV */
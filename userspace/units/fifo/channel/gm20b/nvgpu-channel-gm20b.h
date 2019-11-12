/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef UNIT_NVGPU_CHANNEL_GM20B_H
#define UNIT_NVGPU_CHANNEL_GM20B_H

#include <nvgpu/types.h>

struct unit_module;
struct gk20a;

/** @addtogroup SWUTS-fifo-channel-gm20b
 *  @{
 *
 * Software Unit Test Specification for fifo/channel/gm20b
 */

/**
 * Test specification for: test_gm20b_channel_bind
 *
 * Description: Branch coverage for gm20b_channel_bind
 *
 * Test Type: Feature based
 *
 * Input: test_fifo_init_support() run for this GPU
 *
 * Steps:
 * - Create channel.
 * - Write 0 to ccsr_channel_inst_r and ccsr_channel_r.
 * - Call g->ops.channel.bind.
 * - Check that ccsr_channel_inst_r and ccsr_channel_r have
 *   been programmed.
 * - Check that ch->bound is set.
 *
 * Output: Returns PASS if all branches gave expected results. FAIL otherwise.
 */
int test_gm20b_channel_bind(struct unit_module *m,
		struct gk20a *g, void *args);

/**
 * @}
 */

#endif /* UNIT_NVGPU_CHANNEL_GM20B_H */

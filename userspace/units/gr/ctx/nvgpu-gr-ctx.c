/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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


#include <stdlib.h>

#include <unit/unit.h>
#include <unit/io.h>

#include <nvgpu/posix/io.h>
#include <nvgpu/gk20a.h>
#include <nvgpu/dma.h>
#include <nvgpu/gr/gr.h>
#include <nvgpu/gr/ctx.h>
#include <nvgpu/gr/ctx_mappings.h>

#include <nvgpu/posix/posix-fault-injection.h>
#include <nvgpu/posix/dma.h>

#include "common/gr/gr_priv.h"
#include "common/gr/ctx_priv.h"

#include "../nvgpu-gr.h"
#include "nvgpu-gr-ctx.h"

#define DUMMY_SIZE	0xF0U

static u64 nvgpu_gmmu_map_locked_stub(struct vm_gk20a *vm,
			  u64 vaddr,
			  struct nvgpu_sgt *sgt,
			  u64 buffer_offset,
			  u64 size,
			  u32 pgsz_idx,
			  u8 kind_v,
			  u32 ctag_offset,
			  u32 flags,
			  enum gk20a_mem_rw_flag rw_flag,
			  bool clear_ctags,
			  bool sparse,
			  bool priv,
			  struct vm_gk20a_mapping_batch *batch,
			  enum nvgpu_aperture aperture)
{
	return 1;
}

static void nvgpu_gmmu_unmap_locked_stub(struct vm_gk20a *vm,
			     u64 vaddr,
			     u64 size,
			     u32 pgsz_idx,
			     bool va_allocated,
			     enum gk20a_mem_rw_flag rw_flag,
			     bool sparse,
			     struct vm_gk20a_mapping_batch *batch)
{
	return;
}

int test_gr_ctx_error_injection(struct unit_module *m,
		struct gk20a *g, void *args)
{
	int err;
	struct mm_gk20a *mm = &g->mm;
	struct vm_gk20a *vm;
	struct nvgpu_gr_ctx_desc *desc;
	struct nvgpu_gr_global_ctx_buffer_desc *global_desc;
	struct nvgpu_gr_ctx_mappings *mappings = NULL;
	struct nvgpu_gr_ctx *gr_ctx = NULL;
	struct nvgpu_posix_fault_inj *dma_fi =
		nvgpu_dma_alloc_get_fault_injection();
	struct nvgpu_posix_fault_inj *kmem_fi =
		nvgpu_kmem_get_fault_injection();
	u64 low_hole = SZ_4K * 16UL;
	struct nvgpu_channel *channel = (struct nvgpu_channel *)
		malloc(sizeof(struct nvgpu_channel));
	struct nvgpu_tsg *tsg = (struct nvgpu_tsg *)
		malloc(sizeof(struct nvgpu_tsg));
	u32 i;

	if (channel == NULL || tsg == NULL) {
		unit_return_fail(m, "failed to allocate channel/tsg");
	}

	desc = nvgpu_gr_ctx_desc_alloc(g);
	if (!desc) {
		unit_return_fail(m, "failed to allocate memory");
	}

	vm = nvgpu_vm_init(g, SZ_4K, SZ_4K << 10,
		nvgpu_safe_sub_u64(1ULL << 37, SZ_4K << 10),
		(1ULL << 32), 0ULL,
		false, false, false, "dummy");
	if (!vm) {
		unit_return_fail(m, "failed to allocate VM");
	}

	mm->bar1.aperture_size = 16 << 20;
	mm->bar1.vm = nvgpu_vm_init(g,
			g->ops.mm.gmmu.get_default_big_page_size(),
			low_hole,
			0ULL,
			nvgpu_safe_sub_u64(mm->bar1.aperture_size, low_hole),
			0ULL,
			true, false, false,
			"bar1");
	if (mm->bar1.vm == NULL) {
		unit_return_fail(m, "nvgpu_vm_init failed\n");
	}

	channel->g = g;
	channel->vm = vm;

	g->ops.mm.gmmu.map = nvgpu_gmmu_map_locked_stub;
	g->ops.mm.gmmu.unmap = nvgpu_gmmu_unmap_locked_stub;

	global_desc = nvgpu_gr_global_ctx_desc_alloc(g);
	if (!global_desc) {
		unit_return_fail(m, "failed to allocate desc");
	}

	/* Try to free gr_ctx before it is allocated. */
	nvgpu_gr_ctx_free(g, gr_ctx, NULL);

	gr_ctx = nvgpu_alloc_gr_ctx_struct(g);
	if (!gr_ctx) {
		unit_return_fail(m, "failed to allocate memory");
	}

	tsg->gr_ctx = gr_ctx;

	mappings = nvgpu_gr_ctx_alloc_or_get_mappings(g, tsg, vm);
	if (mappings == NULL) {
		unit_return_fail(m, "failed to allocate gr_ctx mappings");
	}

	/* Context size is not set, so should fail. */
	err = nvgpu_gr_ctx_alloc_ctx_buffers(g, desc, gr_ctx);
	if (err == 0) {
		unit_return_fail(m, "unexpected success");
	}

	/* Set the size now, but inject dma allocation failures. */
	nvgpu_gr_ctx_set_size(desc, NVGPU_GR_CTX_CTX, DUMMY_SIZE);
	nvgpu_gr_ctx_set_size(desc, NVGPU_GR_CTX_PATCH_CTX, DUMMY_SIZE);

	for (i = 0; i < 2; i++) {
		nvgpu_posix_enable_fault_injection(dma_fi, true, i);
		err = nvgpu_gr_ctx_alloc_ctx_buffers(g, desc, gr_ctx);
		if (err == 0) {
			unit_return_fail(m, "unexpected success");
		}
		nvgpu_posix_enable_fault_injection(dma_fi, false, 0);
	}

	err = nvgpu_gr_ctx_alloc_ctx_buffers(g, desc, gr_ctx);
	if (err != 0) {
		unit_return_fail(m, "unexpected success");
	}

	/* Inject kmem alloc failures to trigger mapping failures */
	for (i = 0; i < 2; i++) {
		nvgpu_posix_enable_fault_injection(kmem_fi, true, 2 * i);
		err = nvgpu_gr_ctx_mappings_map_gr_ctx_buffers(g, gr_ctx,
					global_desc, mappings, false);
		if (err == 0) {
			unit_return_fail(m, "unexpected success");
		}
		nvgpu_posix_enable_fault_injection(kmem_fi, false, 0);
	}

	/* global ctx_desc size is not set. */
	err = nvgpu_gr_ctx_mappings_map_gr_ctx_buffers(g, gr_ctx, global_desc,
				       mappings, false);
	if (err == 0) {
		unit_return_fail(m, "unexpected success");
	}

	nvgpu_gr_global_ctx_set_size(global_desc, NVGPU_GR_GLOBAL_CTX_CIRCULAR,
		DUMMY_SIZE);
	nvgpu_gr_global_ctx_set_size(global_desc, NVGPU_GR_GLOBAL_CTX_PAGEPOOL,
		DUMMY_SIZE);
	nvgpu_gr_global_ctx_set_size(global_desc, NVGPU_GR_GLOBAL_CTX_ATTRIBUTE,
		DUMMY_SIZE);
	nvgpu_gr_global_ctx_set_size(global_desc, NVGPU_GR_GLOBAL_CTX_PRIV_ACCESS_MAP,
		DUMMY_SIZE);

	err = nvgpu_gr_global_ctx_buffer_alloc(g, global_desc);
	if (err != 0) {
		unit_return_fail(m, "failed to allocate global buffers");
	}

	/* Fail global ctx buffer mappings */
	for (i = 0; i < 4; i++) {
		nvgpu_posix_enable_fault_injection(kmem_fi, true, 4 + (2 * i));
		err = nvgpu_gr_ctx_mappings_map_gr_ctx_buffers(g, gr_ctx, global_desc,
					       mappings, false);
		if (err == 0) {
			unit_return_fail(m, "unexpected success");
		}
		nvgpu_posix_enable_fault_injection(kmem_fi, false, 0);
	}


	/* Successful mapping */
	err = nvgpu_gr_ctx_mappings_map_gr_ctx_buffers(g, gr_ctx, global_desc,
				       mappings, false);
	if (err != 0) {
		unit_return_fail(m, "failed to map global buffers");
	}

	/* Update the patch buffer */
	nvgpu_gr_ctx_patch_write_begin(g, gr_ctx, true);

	/* Increase data count so that patch write fails */
	gr_ctx->patch_ctx.data_count = 1000;
	nvgpu_gr_ctx_patch_write(g, gr_ctx, 0, 0, true);

	/* Restore data count so that patch write passes */
	gr_ctx->patch_ctx.data_count = 0;
	nvgpu_gr_ctx_patch_write(g, gr_ctx, 0, 0, true);

	/*
	 * Trigger patch write with NULL context, should fail.
	 * We currently don't have API to read contents of patch buffer
	 * hence can't verify yet.
	 */
	nvgpu_gr_ctx_patch_write(g, NULL, 0, 0xDEADBEEF, true);

	nvgpu_gr_ctx_patch_write_end(g, gr_ctx, true);

	/* cleanup */
	nvgpu_gr_ctx_free(g, gr_ctx, global_desc);
	nvgpu_free_gr_ctx_struct(g, gr_ctx);
	nvgpu_gr_ctx_desc_free(g, desc);
	nvgpu_vm_put(g->mm.bar1.vm);

	return UNIT_SUCCESS;
}

struct unit_module_test nvgpu_gr_ctx_tests[] = {
	UNIT_TEST(gr_ctx_setup, test_gr_init_setup, NULL, 0),
	UNIT_TEST(gr_ctx_alloc_errors, test_gr_ctx_error_injection, NULL, 0),
	UNIT_TEST(gr_ctx_cleanup, test_gr_remove_setup, NULL, 0),
};

UNIT_MODULE(nvgpu_gr_ctx, nvgpu_gr_ctx_tests, UNIT_PRIO_NVGPU_TEST);

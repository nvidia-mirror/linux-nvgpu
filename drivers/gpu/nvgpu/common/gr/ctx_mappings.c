/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <nvgpu/gk20a.h>
#include <nvgpu/static_analysis.h>
#include <nvgpu/gr/global_ctx.h>
#include <nvgpu/gr/ctx.h>
#include <nvgpu/gr/ctx_mappings.h>
#include <nvgpu/vm.h>
#include <nvgpu/io.h>
#include <nvgpu/gmmu.h>
#include <nvgpu/dma.h>
#include <nvgpu/string.h>

#include <nvgpu/power_features/pg.h>
#include "common/gr/ctx_mappings_priv.h"

struct nvgpu_gr_ctx_mappings *nvgpu_gr_ctx_mappings_create(struct gk20a *g,
				struct nvgpu_tsg *tsg, struct vm_gk20a *vm)
{
	struct nvgpu_gr_ctx_mappings *mappings = NULL;

	nvgpu_log(g, gpu_dbg_gr, " ");

	if (tsg == NULL || vm == NULL) {
		return NULL;
	}

	mappings = (struct nvgpu_gr_ctx_mappings *)
			nvgpu_kzalloc(g, sizeof(struct nvgpu_gr_ctx_mappings));
	if (mappings == NULL) {
		nvgpu_err(g, "failed to alloc mappings");
		return NULL;
	}

	nvgpu_vm_get(vm);
	mappings->tsg = tsg;
	mappings->vm = vm;

	nvgpu_log(g, gpu_dbg_gr, "done");

	return mappings;
}

void nvgpu_gr_ctx_mappings_free(struct gk20a *g,
				struct nvgpu_gr_ctx_mappings *mappings)
{
	nvgpu_log(g, gpu_dbg_gr, " ");

	nvgpu_vm_put(mappings->vm);
	nvgpu_kfree(g, mappings);

	nvgpu_log(g, gpu_dbg_gr, "done");
}

int nvgpu_gr_ctx_mappings_map_ctx_buffer(struct gk20a *g,
	struct nvgpu_gr_ctx *ctx, u32 index,
	struct nvgpu_gr_ctx_mappings *mappings)
{
	struct vm_gk20a *vm = mappings->vm;
	struct nvgpu_mem *mem;
	u32 mapping_flags;
	u64 gpu_va;

	nvgpu_log(g, gpu_dbg_gr, " ");

	mem = nvgpu_gr_ctx_get_ctx_mem(ctx, index);
	mapping_flags = nvgpu_gr_ctx_get_ctx_mapping_flags(ctx, index);

	nvgpu_assert(mappings->ctx_buffer_va[index] == 0ULL);

	if (nvgpu_mem_is_valid(mem)) {
		gpu_va = nvgpu_gmmu_map(vm,
				mem,
				mapping_flags,
				gk20a_mem_flag_none, true,
				mem->aperture);
		if (gpu_va == 0ULL) {
			nvgpu_err(g, "failed to map ctx buffer %u", index);
			return -ENOMEM;
		}

		mappings->ctx_buffer_va[index] = gpu_va;

		nvgpu_log(g, gpu_dbg_gr, "buffer[%u] mapped at address 0x%llx", index, gpu_va);

#ifdef CONFIG_NVGPU_DEBUGGER
		if (index == NVGPU_GR_CTX_PM_CTX) {
			nvgpu_gr_ctx_set_pm_ctx_mapped(ctx, true);
		}
#endif
	} else {
		nvgpu_log(g, gpu_dbg_gr, "buffer not allocated");
	}

	nvgpu_log(g, gpu_dbg_gr, "done");

	return 0;
}

static void nvgpu_gr_ctx_mappings_unmap_ctx_buffer(struct nvgpu_gr_ctx *ctx,
	u32 index, struct nvgpu_gr_ctx_mappings *mappings)
{
	struct vm_gk20a *vm = mappings->vm;
	struct nvgpu_mem *mem;

	mem = nvgpu_gr_ctx_get_ctx_mem(ctx, index);

	if (nvgpu_mem_is_valid(mem) &&
	    (mappings->ctx_buffer_va[index] != 0ULL)) {
		nvgpu_gmmu_unmap_addr(vm, mem, mappings->ctx_buffer_va[index]);
		mappings->ctx_buffer_va[index] = 0ULL;

#ifdef CONFIG_NVGPU_DEBUGGER
		if (index == NVGPU_GR_CTX_PM_CTX) {
			nvgpu_gr_ctx_set_pm_ctx_mapped(ctx, false);
		}
#endif
	}
}

static void nvgpu_gr_ctx_mappings_unmap_ctx_buffers(struct nvgpu_gr_ctx *ctx,
	struct nvgpu_gr_ctx_mappings *mappings)
{
	u32 i;

	for (i = 0; i < NVGPU_GR_CTX_COUNT; i++) {
		nvgpu_gr_ctx_mappings_unmap_ctx_buffer(ctx, i, mappings);
	}
}

static int nvgpu_gr_ctx_mappings_map_ctx_buffers(struct gk20a *g,
	struct nvgpu_gr_ctx *ctx,
	struct nvgpu_gr_ctx_mappings *mappings)
{
	int err = 0;
	u32 i;

	for (i = 0; i < NVGPU_GR_CTX_COUNT; i++) {
		err = nvgpu_gr_ctx_mappings_map_ctx_buffer(g, ctx, i, mappings);
		if (err != 0) {
			nvgpu_err(g, "gr_ctx buffer %u map failed %d", i, err);
			nvgpu_gr_ctx_mappings_unmap_ctx_buffers(ctx, mappings);
			return err;
		}
	}

	return err;
}

#ifdef CONFIG_NVGPU_GFXP
static void nvgpu_gr_ctx_mappings_unmap_ctx_preemption_buffers(
	struct nvgpu_gr_ctx *ctx,
	struct nvgpu_gr_ctx_mappings *mappings)
{
	u32 i;

	for (i = NVGPU_GR_CTX_PREEMPT_CTXSW;
			i <= NVGPU_GR_CTX_GFXP_RTVCB_CTXSW; i++) {
		nvgpu_gr_ctx_mappings_unmap_ctx_buffer(ctx, i, mappings);
	}
}

int nvgpu_gr_ctx_mappings_map_ctx_preemption_buffers(struct gk20a *g,
	struct nvgpu_gr_ctx *ctx,
	struct nvgpu_gr_ctx_mappings *mappings)
{
	int err = 0;
	u32 i;

	nvgpu_log(g, gpu_dbg_gr, " ");

	for (i = NVGPU_GR_CTX_PREEMPT_CTXSW;
			i <= NVGPU_GR_CTX_GFXP_RTVCB_CTXSW; i++) {
		if (mappings->ctx_buffer_va[i] == 0ULL) {
			err = nvgpu_gr_ctx_mappings_map_ctx_buffer(g, ctx, i, mappings);
			if (err != 0) {
				nvgpu_err(g, "gr_ctx buffer %u map failed %d", i, err);
				nvgpu_gr_ctx_mappings_unmap_ctx_preemption_buffers(ctx, mappings);
				return err;
			}
		}
	}

	nvgpu_log(g, gpu_dbg_gr, "done");

	return err;
}
#endif

static int nvgpu_gr_ctx_mappings_map_global_ctx_buffer(
	struct nvgpu_gr_global_ctx_buffer_desc *global_ctx_buffer,
	u32 va_type, u32 buffer_type, u32 buffer_vpr_type,
	bool vpr, struct nvgpu_gr_ctx_mappings *mappings)
{
	struct vm_gk20a *vm = mappings->vm;
	u64 *g_bfr_va;
	u32 *g_bfr_index;
	u64 gpu_va = 0ULL;

	(void)vpr;
	(void)buffer_vpr_type;

	g_bfr_va = &mappings->global_ctx_buffer_va[0];
	g_bfr_index = &mappings->global_ctx_buffer_index[0];

#ifdef CONFIG_NVGPU_VPR
	if (vpr && nvgpu_gr_global_ctx_buffer_ready(global_ctx_buffer,
					buffer_vpr_type)) {
		gpu_va = nvgpu_gr_global_ctx_buffer_map(global_ctx_buffer,
					buffer_vpr_type,
					vm, true);
		g_bfr_index[va_type] = buffer_vpr_type;
	} else {
#endif
		gpu_va = nvgpu_gr_global_ctx_buffer_map(global_ctx_buffer,
					buffer_type,
					vm, true);
		g_bfr_index[va_type] = buffer_type;
#ifdef CONFIG_NVGPU_VPR
	}
#endif
	if (gpu_va == 0ULL) {
		goto clean_up;
	}

	g_bfr_va[va_type] = gpu_va;

	return 0;

clean_up:
	return -ENOMEM;
}

static void nvgpu_gr_ctx_mappings_unmap_global_ctx_buffers(
	struct nvgpu_gr_global_ctx_buffer_desc *global_ctx_buffer,
	struct nvgpu_gr_ctx_mappings *mappings)
{
	u64 *g_bfr_va = &mappings->global_ctx_buffer_va[0];
	u32 *g_bfr_index = &mappings->global_ctx_buffer_index[0];
	struct vm_gk20a *vm = mappings->vm;
	u32 i;

	for (i = 0U; i < NVGPU_GR_GLOBAL_CTX_VA_COUNT; i++) {
		if (g_bfr_va[i] != 0ULL) {
			nvgpu_gr_global_ctx_buffer_unmap(global_ctx_buffer,
				g_bfr_index[i], vm, g_bfr_va[i]);
		}
	}

	(void) memset(g_bfr_va, 0, sizeof(mappings->global_ctx_buffer_va));
	(void) memset(g_bfr_index, 0, sizeof(mappings->global_ctx_buffer_index));
}

static int nvgpu_gr_ctx_mappings_map_global_ctx_buffers(struct gk20a *g,
	struct nvgpu_gr_global_ctx_buffer_desc *global_ctx_buffer,
	struct nvgpu_gr_ctx_mappings *mappings, bool vpr)
{
	int err;

	/*
	 * MIG supports only compute class.
	 * Allocate BUNDLE_CB, PAGEPOOL, ATTRIBUTE_CB and RTV_CB
	 * if 2D/3D/I2M classes(graphics) are supported.
	 */
	if (!nvgpu_is_enabled(g, NVGPU_SUPPORT_MIG)) {
		/* Circular Buffer */
		err = nvgpu_gr_ctx_mappings_map_global_ctx_buffer(
					global_ctx_buffer,
					NVGPU_GR_GLOBAL_CTX_CIRCULAR_VA,
					NVGPU_GR_GLOBAL_CTX_CIRCULAR,
#ifdef CONFIG_NVGPU_VPR
					NVGPU_GR_GLOBAL_CTX_CIRCULAR_VPR,
#else
					NVGPU_GR_GLOBAL_CTX_CIRCULAR,
#endif
					vpr, mappings);
		if (err != 0) {
			nvgpu_err(g, "cannot map ctx circular buffer");
			goto fail;
		}

		/* Attribute Buffer */
		err = nvgpu_gr_ctx_mappings_map_global_ctx_buffer(
					global_ctx_buffer,
					NVGPU_GR_GLOBAL_CTX_ATTRIBUTE_VA,
					NVGPU_GR_GLOBAL_CTX_ATTRIBUTE,
#ifdef CONFIG_NVGPU_VPR
					NVGPU_GR_GLOBAL_CTX_ATTRIBUTE_VPR,
#else
					NVGPU_GR_GLOBAL_CTX_ATTRIBUTE,
#endif
					vpr, mappings);
		if (err != 0) {
			nvgpu_err(g, "cannot map ctx attribute buffer");
			goto fail;
		}

		/* Page Pool */
		err = nvgpu_gr_ctx_mappings_map_global_ctx_buffer(
					global_ctx_buffer,
					NVGPU_GR_GLOBAL_CTX_PAGEPOOL_VA,
					NVGPU_GR_GLOBAL_CTX_PAGEPOOL,
#ifdef CONFIG_NVGPU_VPR
					NVGPU_GR_GLOBAL_CTX_PAGEPOOL_VPR,
#else
					NVGPU_GR_GLOBAL_CTX_PAGEPOOL,
#endif
					vpr, mappings);
		if (err != 0) {
			nvgpu_err(g, "cannot map ctx pagepool buffer");
			goto fail;
		}
#ifdef CONFIG_NVGPU_GRAPHICS
		/*
		 * RTV circular buffer. Note that this is non-VPR buffer always.
		 */
		if (nvgpu_gr_global_ctx_buffer_ready(global_ctx_buffer,
				NVGPU_GR_GLOBAL_CTX_RTV_CIRCULAR_BUFFER)) {
			err  = nvgpu_gr_ctx_mappings_map_global_ctx_buffer(
					global_ctx_buffer,
					NVGPU_GR_GLOBAL_CTX_RTV_CIRCULAR_BUFFER_VA,
					NVGPU_GR_GLOBAL_CTX_RTV_CIRCULAR_BUFFER,
					NVGPU_GR_GLOBAL_CTX_RTV_CIRCULAR_BUFFER,
					false, mappings);
			if (err != 0) {
				nvgpu_err(g,
					"cannot map ctx rtv circular buffer");
				goto fail;
			}
		}
#endif
	}

	/* Priv register Access Map. Note that this is non-VPR buffer always. */
	err  = nvgpu_gr_ctx_mappings_map_global_ctx_buffer(
			global_ctx_buffer,
			NVGPU_GR_GLOBAL_CTX_PRIV_ACCESS_MAP_VA,
			NVGPU_GR_GLOBAL_CTX_PRIV_ACCESS_MAP,
			NVGPU_GR_GLOBAL_CTX_PRIV_ACCESS_MAP,
			false, mappings);
	if (err != 0) {
		nvgpu_err(g, "cannot map ctx priv access buffer");
		goto fail;
	}

#ifdef CONFIG_NVGPU_FECS_TRACE
	/* FECS trace buffer. Note that this is non-VPR buffer always. */
	if (nvgpu_is_enabled(g, NVGPU_FECS_TRACE_VA)) {
		err  = nvgpu_gr_ctx_mappings_map_global_ctx_buffer(
			global_ctx_buffer,
			NVGPU_GR_GLOBAL_CTX_FECS_TRACE_BUFFER_VA,
			NVGPU_GR_GLOBAL_CTX_FECS_TRACE_BUFFER,
			NVGPU_GR_GLOBAL_CTX_FECS_TRACE_BUFFER,
			false, mappings);
		if (err != 0) {
			nvgpu_err(g, "cannot map ctx fecs trace buffer");
			goto fail;
		}
	}
#endif

	return 0;

fail:
	nvgpu_gr_ctx_mappings_unmap_global_ctx_buffers(
		global_ctx_buffer, mappings);
	return err;
}

int nvgpu_gr_ctx_mappings_map_gr_ctx_buffers(struct gk20a *g,
	struct nvgpu_gr_ctx *gr_ctx,
	struct nvgpu_gr_global_ctx_buffer_desc *global_ctx_buffer,
	struct nvgpu_gr_ctx_mappings *mappings,
	bool vpr)
{
	int err;

	nvgpu_log(g, gpu_dbg_gr, " ");

	if (gr_ctx == NULL || global_ctx_buffer == NULL ||
	    mappings == NULL) {
		nvgpu_err(g, "mappings/gr_ctx/global_ctx_buffer struct null");
		return -EINVAL;
	}

	err = nvgpu_gr_ctx_mappings_map_ctx_buffers(g, gr_ctx, mappings);
	if (err != 0) {
		nvgpu_err(g, "fail to map ctx buffers");
		return err;
	}

	err = nvgpu_gr_ctx_mappings_map_global_ctx_buffers(g,
			global_ctx_buffer, mappings, vpr);
	if (err != 0) {
		nvgpu_err(g, "fail to map global ctx buffer");
		nvgpu_gr_ctx_mappings_unmap_ctx_buffers(gr_ctx, mappings);
		return err;
	}

	nvgpu_log(g, gpu_dbg_gr, "done");

	return err;
}

void nvgpu_gr_ctx_unmap_buffers(struct gk20a *g,
	struct nvgpu_gr_ctx *gr_ctx,
	struct nvgpu_gr_global_ctx_buffer_desc *global_ctx_buffer,
	struct nvgpu_gr_ctx_mappings *mappings)
{
	nvgpu_log(g, gpu_dbg_gr, " ");

	nvgpu_gr_ctx_mappings_unmap_global_ctx_buffers(global_ctx_buffer,
		mappings);

	nvgpu_gr_ctx_mappings_unmap_ctx_buffers(gr_ctx, mappings);

	nvgpu_log(g, gpu_dbg_gr, "done");
}

u64 nvgpu_gr_ctx_mappings_get_global_ctx_va(struct nvgpu_gr_ctx_mappings *mappings,
	u32 index)
{
	nvgpu_assert(index < NVGPU_GR_GLOBAL_CTX_VA_COUNT);
	return mappings->global_ctx_buffer_va[index];
}

u64 nvgpu_gr_ctx_mappings_get_ctx_va(struct nvgpu_gr_ctx_mappings *mappings,
	u32 index)
{
	nvgpu_assert(index < NVGPU_GR_CTX_COUNT);
	return mappings->ctx_buffer_va[index];
}

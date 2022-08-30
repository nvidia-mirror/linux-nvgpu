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

#include <unistd.h>

#include <unit/unit.h>
#include <unit/io.h>
#include <unit/utils.h>

#include <nvgpu/types.h>
#include <nvgpu/gk20a.h>
#include <nvgpu/channel.h>
#include <nvgpu/runlist.h>
#include <nvgpu/tsg.h>
#include <nvgpu/class.h>
#include <nvgpu/falcon.h>

#include <nvgpu/gr/gr.h>
#include <nvgpu/gr/ctx.h>
#include <nvgpu/gr/obj_ctx.h>

#include <nvgpu/hw/gv11b/hw_gr_gv11b.h>

#include <nvgpu/posix/io.h>
#include <nvgpu/posix/kmem.h>
#include <nvgpu/posix/dma.h>
#include <nvgpu/posix/posix-fault-injection.h>
#include <nvgpu/string.h>
#include <nvgpu/gr/subctx.h>

#include "common/gr/gr_priv.h"
#include "common/gr/obj_ctx_priv.h"
#include "common/gr/ctx_priv.h"
#include "common/gr/ctx_mappings_priv.h"
#include "common/fifo/tsg_subctx_priv.h"
#include "common/gr/subctx_priv.h"

#include "../nvgpu-gr.h"
#include "nvgpu-gr-setup.h"

#define CLASS_MIN_VALUE 	0
#define CLASS_MAX_VALUE 	U32_MAX
#define CLASS_VALID_VALUE 	0x1234

#define FLAGS_MIN_VALUE 	0
#define FLAGS_MAX_VALUE 	U32_MAX
#define FLAGS_VALID_VALUE 	0x1234

struct gr_gops_org {
	int (*l2_flush)(struct gk20a *g, bool invalidate);
	int (*fe_pwr_mode)(struct gk20a *g, bool force_on);
	int (*wait_idle)(struct gk20a *g);
	int (*ctrl_ctxsw)(struct gk20a *g, u32 fecs_method,
			u32 data, u32 *ret_val);
	int (*fifo_preempt_tsg)(struct gk20a *g,
				struct nvgpu_tsg *tsg);
	bool (*is_valid)(u32 class_num);
	bool (*is_valid_compute)(u32 class_num);
};

static struct nvgpu_channel *gr_setup_ch;
static struct nvgpu_tsg *gr_setup_tsg;
static struct gr_gops_org gr_setup_gops;

static struct nvgpu_channel **subctx_chs;
static struct gk20a_as_share **subctx_as_shares;
static struct nvgpu_tsg *subctx_tsg;
static struct gk20a_as_share *shared_subctx_as_share;

static bool stub_class_is_valid(u32 class_num)
{
	return true;
}

static bool stub_class_is_valid_compute(u32 class_num)
{
	return true;
}

static int stub_runlist_update(struct gk20a *g,
			       struct nvgpu_runlist *rl,
			       struct nvgpu_channel *ch,
			       bool add, bool wait_for_finish)
{
	return 0;
}

static int stub_mm_l2_flush(struct gk20a *g, bool invalidate)
{
	return 0;
}

static int stub_gr_init_fe_pwr_mode(struct gk20a *g, bool force_on)
{
	return 0;
}

static int stub_gr_init_wait_idle(struct gk20a *g)
{
	return 0;
}

static int stub_gr_falcon_ctrl_ctxsw(struct gk20a *g, u32 fecs_method,
				     u32 data, u32 *ret_val)
{
	return 0;
}

static int stub_gr_fifo_preempt_tsg(struct gk20a *g, struct nvgpu_tsg *tsg)
{
	return -1;
}

static void gr_setup_stub_class_ops(struct gk20a *g)
{
	g->ops.gpu_class.is_valid = stub_class_is_valid;
	g->ops.gpu_class.is_valid_compute = stub_class_is_valid_compute;
}

static void gr_setup_restore_class_ops(struct gk20a *g)
{
	g->ops.gpu_class.is_valid =
		gr_setup_gops.is_valid;
	g->ops.gpu_class.is_valid_compute =
		gr_setup_gops.is_valid_compute;
}

static void gr_setup_save_class_ops(struct gk20a *g)
{
	gr_setup_gops.is_valid =
		g->ops.gpu_class.is_valid;
	gr_setup_gops.is_valid_compute =
		g->ops.gpu_class.is_valid_compute;
}

static int gr_test_setup_unbind_tsg(struct unit_module *m, struct gk20a *g)
{
	int err = 0;

	if ((gr_setup_ch == NULL) || (gr_setup_tsg == NULL)) {
		goto unbind_tsg;
	}

	err = nvgpu_tsg_unbind_channel(gr_setup_tsg, gr_setup_ch, true);
	if (err != 0) {
		unit_err(m, "failed tsg channel unbind\n");
	}

unbind_tsg:
	return (err == 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

static void gr_test_setup_cleanup_ch_tsg(struct unit_module *m,
					 struct gk20a *g)
{
	if (gr_setup_ch != NULL) {
		nvgpu_channel_close(gr_setup_ch);
	}

	if (gr_setup_tsg != NULL) {
		nvgpu_ref_put(&gr_setup_tsg->refcount, nvgpu_tsg_release);
	}

	gr_setup_tsg = NULL;
	gr_setup_ch = NULL;
}

static int gr_test_setup_allocate_ch_tsg(struct unit_module *m,
					 struct gk20a *g)
{
	u32 tsgid = getpid();
	struct nvgpu_channel *ch = NULL;
	struct nvgpu_tsg *tsg = NULL;
        struct gk20a_as_share *as_share = NULL;
	int err;

	err = nvgpu_channel_setup_sw(g);
	if (err != 0) {
		unit_return_fail(m, "failed channel setup\n");
	}

	err = nvgpu_tsg_setup_sw(g);
	if (err != 0) {
		unit_return_fail(m, "failed tsg setup\n");
	}

	tsg = nvgpu_tsg_open(g, tsgid);
	if (tsg == NULL) {
		unit_return_fail(m, "failed tsg open\n");
	}

	ch = nvgpu_channel_open_new(g, NVGPU_INVALID_RUNLIST_ID,
				false, tsgid, tsgid);
	if (ch == NULL) {
		unit_err(m, "failed channel open\n");
		goto ch_cleanup;
	}

	err = gk20a_as_alloc_share(g,
		0U, NVGPU_AS_ALLOC_UNIFIED_VA,
		U64(SZ_4K) << U64(10),
		(1ULL << 37), 0ULL, &as_share);
	if (err != 0) {
		unit_err(m, "failed vm memory alloc\n");
		goto tsg_unbind;
	}

	err = g->ops.mm.vm_bind_channel(as_share->vm, ch);
	if (err != 0) {
		unit_err(m, "failed vm binding to ch\n");
		goto tsg_unbind;
	}

	err = nvgpu_tsg_bind_channel(tsg, ch);
	if (err != 0) {
		unit_err(m, "failed tsg channel bind\n");
		goto ch_cleanup;
	}

	gr_setup_ch = ch;
	gr_setup_tsg = tsg;

	goto ch_alloc_end;

tsg_unbind:
	gr_test_setup_unbind_tsg(m, g);

ch_cleanup:
	gr_test_setup_cleanup_ch_tsg(m, g);

ch_alloc_end:
	return (err == 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

static int gr_test_setup_free_subctx_ch_tsg(struct unit_module *m,
					 struct gk20a *g)
{
	u32 max_subctx_count = g->ops.gr.init.get_max_subctx_count();
	u32 i;

	for (i = 0; i < max_subctx_count; i++) {
		if (subctx_chs[i] != NULL) {
			nvgpu_channel_close(subctx_chs[i]);
			subctx_chs[i] = NULL;
		}

		if (subctx_as_shares && subctx_as_shares[i]) {
			gk20a_as_release_share(subctx_as_shares[i]);
			subctx_as_shares[i] = NULL;
		}
	}

	nvgpu_kfree(g, subctx_chs);
	subctx_chs = NULL;

	if (shared_subctx_as_share) {
		gk20a_as_release_share(shared_subctx_as_share);
		shared_subctx_as_share = NULL;
	}

	if (!nvgpu_list_empty(&subctx_tsg->gr_ctx_mappings_list)) {
		unit_err(m, "mappings not freed");
		return UNIT_FAIL;
	}

	if (subctx_tsg != NULL) {
		nvgpu_ref_put(&subctx_tsg->refcount, nvgpu_tsg_release);
		subctx_tsg = NULL;
	}

	nvgpu_kfree(g, subctx_as_shares);
	subctx_as_shares = NULL;

	return UNIT_SUCCESS;
}

static int gr_test_setup_allocate_subctx_ch_tsg(struct unit_module *m,
					 struct gk20a *g, bool shared_vm)
{
	u32 max_subctx_count = g->ops.gr.init.get_max_subctx_count();
	u32 tsgid = getpid();
	struct nvgpu_channel *ch = NULL;
	struct nvgpu_tsg *tsg = NULL;
	struct gk20a_as_share *as_share = NULL;
	int err;
	u32 i;

	subctx_chs = (struct nvgpu_channel **)nvgpu_kzalloc(g,
				sizeof(struct nvgpu_channel *) * max_subctx_count);
	if (subctx_chs == NULL) {
		unit_err(m, "failed to alloc subctx chs\n");
		goto cleanup;
	}

	subctx_as_shares = (struct gk20a_as_share **)nvgpu_kzalloc(g,
				sizeof(struct gk20a_as_share *) * max_subctx_count);
	if (subctx_as_shares == NULL) {
		unit_err(m, "failed to alloc subctx as shares\n");
		goto cleanup;
	}

	tsg = nvgpu_tsg_open(g, tsgid);
	if (tsg == NULL) {
		unit_err(m, "failed tsg open\n");
		goto cleanup;
	}

	subctx_tsg = tsg;

	if (shared_vm) {
		err = gk20a_as_alloc_share(g,
			0U, NVGPU_AS_ALLOC_UNIFIED_VA,
			U64(SZ_4K) << U64(10),
			(1ULL << 37), 0ULL, &as_share);
		if (err != 0) {
			unit_err(m, "failed vm memory alloc\n");
			goto tsg_cleanup;
		}

		shared_subctx_as_share = as_share;
	}

	for (i = 0; i < max_subctx_count; i++) {
		ch = nvgpu_channel_open_new(g, NVGPU_INVALID_RUNLIST_ID,
					false, tsgid, tsgid);
		if (ch == NULL) {
			unit_err(m, "failed channel open\n");
			goto ch_cleanup;
		}

		subctx_chs[i] = ch;

		if (shared_vm) {
			as_share = shared_subctx_as_share;
		} else {
			err = gk20a_as_alloc_share(g,
				0U, NVGPU_AS_ALLOC_UNIFIED_VA,
				U64(SZ_4K) << U64(10),
				(1ULL << 37), 0ULL, &subctx_as_shares[i]);
			if (err != 0) {
				unit_err(m, "failed vm memory alloc\n");
				goto ch_cleanup;
			}

			as_share = subctx_as_shares[i];
		}

		err = g->ops.mm.vm_bind_channel(as_share->vm, ch);
		if (err != 0) {
			unit_err(m, "failed vm binding to ch\n");
			goto ch_cleanup;
		}

		ch->subctx_id = i;

		err = nvgpu_tsg_bind_channel(tsg, ch);
		if (err != 0) {
			unit_err(m, "failed tsg channel bind\n");
			goto ch_cleanup;
		}

		err = g->ops.gr.setup.alloc_obj_ctx(ch, VOLTA_COMPUTE_A, 0);
		if (err != 0) {
			unit_err(m, "setup alloc obj_ctx failed\n");
			goto ch_cleanup;
		}
	}

	goto ch_alloc_end;

ch_cleanup:
	gr_test_setup_free_subctx_ch_tsg(m, g);
	goto ch_alloc_end;

tsg_cleanup:
	if (subctx_tsg != NULL) {
		nvgpu_ref_put(&subctx_tsg->refcount, nvgpu_tsg_release);
		subctx_tsg = NULL;
	}

cleanup:
	nvgpu_kfree(g, subctx_as_shares);
	nvgpu_kfree(g, subctx_chs);

ch_alloc_end:
	return (err == 0) ? UNIT_SUCCESS : UNIT_FAIL;
}

static int gr_test_setup_compare_mappings(struct unit_module *m,
				struct nvgpu_gr_ctx_mappings *veid0_mappings,
				struct nvgpu_gr_ctx_mappings *mappings)
{
	if (nvgpu_memcmp((u8 *)&veid0_mappings->ctx_buffer_va,
			 (u8 *)&mappings->ctx_buffer_va,
			 NVGPU_GR_CTX_COUNT * sizeof(u64)) != 0) {
		unit_err(m, "ctx buffer va mismatch\n");
		return UNIT_FAIL;
	}

	if (nvgpu_memcmp((u8 *)&veid0_mappings->global_ctx_buffer_va,
			 (u8 *)&mappings->global_ctx_buffer_va,
			 NVGPU_GR_GLOBAL_CTX_VA_COUNT * sizeof(u64)) != 0) {
		unit_err(m, "global ctx buffer va mismatch\n");
		return UNIT_FAIL;
	}

	return UNIT_SUCCESS;
}

static int gr_test_setup_compare_ctx_headers(struct unit_module *m,
				struct gk20a *g,
				struct nvgpu_mem *veid0_subctx_header,
				struct nvgpu_mem *subctx_header)
{
	u32 size = g->ops.gr.ctxsw_prog.hw_get_fecs_header_size();
	u8 *header1 = NULL;
	u8 *header2 = NULL;
	int ret = 0;

	header1 = (u8 *) nvgpu_kzalloc(g, size);
	if (header1 == NULL) {
		unit_err(m, "header1 allocation failed");
		return UNIT_FAIL;
	}

	header2 = (u8 *) nvgpu_kzalloc(g, size);
	if (header2 == NULL) {
		unit_err(m, "header2 allocation failed");
		ret = UNIT_FAIL;
		goto out;
	}

	nvgpu_mem_rd_n(g, veid0_subctx_header, 0, (void *)header1, size);
	nvgpu_mem_rd_n(g, subctx_header, 0, (void *)header2, size);

	if (nvgpu_memcmp(header1, header2, size) != 0) {
		unit_err(m, "subctx header mismatch\n");
		ret = UNIT_FAIL;
		goto out;
	}

out:
	nvgpu_kfree(g, header1);
	nvgpu_kfree(g, header2);
	return ret;
}

static inline struct nvgpu_gr_ctx_mappings *
nvgpu_gr_ctx_mappings_from_tsg_entry(struct nvgpu_list_node *node)
{
	return (struct nvgpu_gr_ctx_mappings *)
	   ((uintptr_t)node - offsetof(struct nvgpu_gr_ctx_mappings, tsg_entry));
};

static inline struct nvgpu_tsg_subctx *
nvgpu_tsg_subctx_from_tsg_entry(struct nvgpu_list_node *node)
{
	return (struct nvgpu_tsg_subctx *)
	   ((uintptr_t)node - offsetof(struct nvgpu_tsg_subctx, tsg_entry));
};

int test_gr_validate_subctx_gr_ctx_buffers(struct unit_module *m,
					 struct gk20a *g, void *args)
{
	u32 max_subctx_count = g->ops.gr.init.get_max_subctx_count();
	struct nvgpu_gr_ctx_mappings *veid0_mappings;
	struct nvgpu_tsg_subctx *subctx;
	struct nvgpu_mem *ctxheader1;
	struct nvgpu_mem *ctxheader2;
	bool shared_vm = true;
	u32 close_ch;
	int err;

	err = gr_test_setup_allocate_subctx_ch_tsg(m, g, shared_vm);
	if (err != 0) {
		unit_return_fail(m, "alloc setup subctx channels failed\n");
	}

	/*
	 * Close any random Async channel to check that it does not change the
	 * state of other channels/subcontexts.
	 */
	srand(time(0));
	close_ch = get_random_u32(1, max_subctx_count - 1U);
	nvgpu_channel_close(subctx_chs[close_ch]);
	subctx_chs[close_ch] = NULL;

	if (nvgpu_list_first_entry(&subctx_tsg->gr_ctx_mappings_list, nvgpu_gr_ctx_mappings,
				   tsg_entry) !=
	    nvgpu_list_last_entry(&subctx_tsg->gr_ctx_mappings_list, nvgpu_gr_ctx_mappings,
				   tsg_entry)) {
		unit_err(m, "Only single element should be present in the"
			 "gr_ctx_mappings_list");
		err = -EINVAL;
		goto out;
	}

	veid0_mappings = subctx_chs[0]->subctx->gr_subctx->mappings;
	if (veid0_mappings == NULL) {
		unit_err(m, "veid0 mappings not initialized\n");
		err = -EINVAL;
		goto out;
	}

	if ((veid0_mappings->global_ctx_buffer_va[NVGPU_GR_GLOBAL_CTX_CIRCULAR_VA] == 0ULL) ||
	    (veid0_mappings->global_ctx_buffer_va[NVGPU_GR_GLOBAL_CTX_PAGEPOOL_VA] == 0ULL) ||
	    (veid0_mappings->global_ctx_buffer_va[NVGPU_GR_GLOBAL_CTX_ATTRIBUTE_VA] == 0ULL)) {
		unit_err(m, "Global ctx buffers not mapped for VEID0");
		err = -EINVAL;
		goto out;
	}

	ctxheader1 = nvgpu_gr_subctx_get_ctx_header(subctx_chs[0]->subctx->gr_subctx);

	nvgpu_list_for_each_entry(subctx, &subctx_tsg->subctx_list,
				  nvgpu_tsg_subctx, tsg_entry) {
		if (subctx->gr_subctx == NULL) {
			unit_err(m, "gr_subctx not initialized\n");
			err = -EINVAL;
			goto out;
		}

		if (subctx->subctx_id == CHANNEL_INFO_VEID0) {
			continue;
		}

		err = gr_test_setup_compare_mappings(m, veid0_mappings,
				subctx->gr_subctx->mappings);
		if (err != 0) {
			unit_err(m, "gr ctx mapping not valid\n");
			err = -EINVAL;
			goto out;
		}

		ctxheader2 = nvgpu_gr_subctx_get_ctx_header(subctx->gr_subctx);

		err = gr_test_setup_compare_ctx_headers(m, g, ctxheader1, ctxheader2);
		if (err != 0) {
			unit_err(m, "gr subctx headers not valid\n");
			err = -EINVAL;
			goto out;
		}
	}

out:
	err = gr_test_setup_free_subctx_ch_tsg(m, g);

	return (err == 0) ? UNIT_SUCCESS : UNIT_FAIL;
}

static void gr_setup_restore_valid_ops(struct gk20a *g)
{
	g->ops.mm.cache.l2_flush =
		gr_setup_gops.l2_flush;
	g->ops.gr.init.fe_pwr_mode_force_on =
		gr_setup_gops.fe_pwr_mode;
	g->ops.gr.init.wait_idle =
		gr_setup_gops.wait_idle;
	g->ops.gr.falcon.ctrl_ctxsw =
		gr_setup_gops.ctrl_ctxsw;
	g->ops.fifo.preempt_tsg =
		gr_setup_gops.fifo_preempt_tsg;
}

static void gr_setup_save_valid_ops(struct gk20a *g)
{
	gr_setup_gops.l2_flush =
		g->ops.mm.cache.l2_flush;
	gr_setup_gops.fe_pwr_mode =
		g->ops.gr.init.fe_pwr_mode_force_on;
	gr_setup_gops.wait_idle =
		g->ops.gr.init.wait_idle;
	gr_setup_gops.ctrl_ctxsw =
		g->ops.gr.falcon.ctrl_ctxsw;
	gr_setup_gops.fifo_preempt_tsg =
		g->ops.fifo.preempt_tsg;
}

static void gr_setup_stub_valid_ops(struct gk20a *g)
{
	g->ops.mm.cache.l2_flush = stub_mm_l2_flush;
	g->ops.gr.init.fe_pwr_mode_force_on = stub_gr_init_fe_pwr_mode;
	g->ops.gr.init.wait_idle = stub_gr_init_wait_idle;
	g->ops.gr.falcon.ctrl_ctxsw = stub_gr_falcon_ctrl_ctxsw;
}

struct test_gr_setup_preemption_mode {
	u32 compute_mode;
	u32 graphics_mode;
	int result;
};

struct test_gr_setup_preemption_mode preemp_mode_types[] = {
	[0] = {
		.compute_mode = NVGPU_PREEMPTION_MODE_COMPUTE_WFI,
		.graphics_mode = 0,
		.result = 0,
	      },
	[1] = {
		.compute_mode = NVGPU_PREEMPTION_MODE_COMPUTE_CTA,
		.graphics_mode = 0,
		.result = 0,
	      },
	[2] = {
		.compute_mode = BIT(15),
		.graphics_mode = 0,
		.result = -EINVAL,
	      },
	[3] = {
		.compute_mode = 0,
		.graphics_mode = 0,
		.result = 0,
	      },
	[4] = {
		.compute_mode = 0,
		.graphics_mode = BIT(0),
		.result = -EINVAL,
	      },
	[5] = {
		.compute_mode = NVGPU_PREEMPTION_MODE_COMPUTE_CTA,
		.graphics_mode = BIT(12),
		.result = -EINVAL,
	      },
	[6] = {
		.compute_mode = NVGPU_PREEMPTION_MODE_COMPUTE_CTA,
		.graphics_mode = U32_MAX,
		.result = -EINVAL,
	      },
	[7] = {
		.compute_mode = 3,
		.graphics_mode = 0,
		.result = -EINVAL,
	      },
	[8] = {
		.compute_mode = U32_MAX,
		.graphics_mode = 0,
		.result = -EINVAL,
	      },
};

int test_gr_setup_preemption_mode_errors(struct unit_module *m,
				      struct gk20a *g, void *args)
{
	int err, i;
	u32 class_num, tsgid;
	int arry_cnt = sizeof(preemp_mode_types)/
			sizeof(struct test_gr_setup_preemption_mode);

	if (gr_setup_ch == NULL) {
		unit_return_fail(m, "Failed setup for valid channel\n");
	}

	/* Various compute and grahics mode for error injection */
	for (i = 0; i < arry_cnt; i++) {
		err = g->ops.gr.setup.set_preemption_mode(gr_setup_ch,
				preemp_mode_types[i].graphics_mode,
				preemp_mode_types[i].compute_mode, 0);
		if (err != preemp_mode_types[i].result) {
			unit_return_fail(m, "Fail Preemp_mode Error Test-1\n");
		}
	}

	/* disable preempt_tsg for failure */
	gr_setup_tsg->gr_ctx->compute_preempt_mode =
			NVGPU_PREEMPTION_MODE_COMPUTE_WFI;
	g->ops.fifo.preempt_tsg = stub_gr_fifo_preempt_tsg;
	err = g->ops.gr.setup.set_preemption_mode(gr_setup_ch, 0,
				NVGPU_PREEMPTION_MODE_COMPUTE_CTA, 0);
	if (err == 0) {
		unit_return_fail(m, "Fail Preemp_mode Error Test-2\n");
	}

	class_num = gr_setup_ch->obj_class;
	tsgid = gr_setup_ch->tsgid;
	/* Unset the tsgid */
	gr_setup_ch->tsgid = NVGPU_INVALID_TSG_ID;
	err = g->ops.gr.setup.set_preemption_mode(gr_setup_ch, 0, 0, 0);
	if (err == 0) {
		unit_return_fail(m, "Fail Preemp_mode Error Test-2\n");
	}

	gr_setup_ch->tsgid = tsgid;
	/* Unset the valid Class*/
	gr_setup_ch->obj_class = 0;
	err = g->ops.gr.setup.set_preemption_mode(gr_setup_ch, 0, 0, 0);
	if (err == 0) {
		unit_return_fail(m, "Fail Preemp_mode Error Test-2\n");
	}

	/* Set invalid Class*/
	gr_setup_ch->obj_class = 0x1234;
	err = g->ops.gr.setup.set_preemption_mode(gr_setup_ch, 0, 0, 0);
	if (err == 0) {
		unit_return_fail(m, "Fail Preemp_mode Error Test-2\n");
	}

	gr_setup_ch->obj_class = class_num;

	return UNIT_SUCCESS;
}

static int gr_setup_fail_subctx_alloc(struct gk20a *g)
{
	int err;
	struct nvgpu_posix_fault_inj *kmem_fi =
		nvgpu_kmem_get_fault_injection();
	struct nvgpu_posix_fault_inj *dma_fi =
		nvgpu_dma_alloc_get_fault_injection();

	/* Alloc Failure in nvgpu_gr_subctx_alloc */
	/* Fail 1 - dma alloc */
	nvgpu_posix_enable_fault_injection(dma_fi, true, 0);
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	if (err == 0) {
		goto sub_ctx_fail_end;
	}
	nvgpu_posix_enable_fault_injection(dma_fi, false, 0);

	/* Fail 2 - kmem alloc */
	nvgpu_posix_enable_fault_injection(kmem_fi, true, 0);
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	if (err == 0) {
		goto sub_ctx_fail_end;
	}
	nvgpu_posix_enable_fault_injection(kmem_fi, false, 0);

	/* Fail 3 - gmmap */
	nvgpu_posix_enable_fault_injection(kmem_fi, true, 1);
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);

sub_ctx_fail_end:
	nvgpu_posix_enable_fault_injection(kmem_fi, false, 0);
	nvgpu_posix_enable_fault_injection(dma_fi, false, 0);
	return (err != 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

static int gr_setup_fail_alloc(struct unit_module *m, struct gk20a *g)
{
	int err;
	u32 tsgid;
	struct vm_gk20a *vm;

	tsgid = gr_setup_ch->tsgid;
	vm = gr_setup_ch->vm;

	/* SUBTEST-1 for invalid tsgid*/
	gr_setup_ch->tsgid = NVGPU_INVALID_TSG_ID;
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	gr_setup_ch->tsgid = tsgid;
	if (err == 0) {
		unit_err(m, "setup alloc SUBTEST-1 failed\n");
		goto obj_ctx_fail_end;
	}

	/* SUBTEST-2 for invalid class num*/
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, 0, 0);
	if (err == 0) {
		unit_err(m, "setup alloc SUBTEST-2 failed\n");
		goto obj_ctx_fail_end;
	}

	/* SUBTEST-3 for invalid channel vm*/
	gr_setup_ch->vm = NULL;
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, 0, 0);
	gr_setup_ch->vm = vm;
	if (err == 0) {
		unit_err(m, "setup alloc SUBTEST-3 failed\n");
		goto obj_ctx_fail_end;
	}

	/* SUBTEST-4 for graphics class num */
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, 0xC397U, 0);
	if (err == 0) {
		unit_err(m, "setup alloc SUBTEST-4 failed\n");
		goto obj_ctx_fail_end;
	}

obj_ctx_fail_end:
	return (err != 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

static int gr_setup_alloc_fail_golden_size(struct unit_module *m, struct gk20a *g)
{
	int err;

	/* Reset golden image size*/
	g->gr->golden_image->size = 0;

	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	if (err == 0) {
		unit_err(m, "setup alloc reset golden size failed\n");
	}

	return (err != 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

static int gr_setup_alloc_fail_fe_pwr_mode(struct unit_module *m, struct gk20a *g)
{
	int err;

	g->ops.mm.cache.l2_flush = stub_mm_l2_flush;

	/* Reset golden image ready bit */
	g->gr->golden_image->ready = false;

	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	if (err == 0) {
		unit_err(m, "setup alloc fe_pwr_mode failed\n");
	}

	return (err != 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

static int gr_setup_alloc_fail_ctrl_ctxsw(struct unit_module *m,
					  struct gk20a *g, void *args)
{
	int err;

	err = gr_test_setup_allocate_ch_tsg(m, g);
	if (err != 0) {
		unit_return_fail(m, "alloc setup channel failed\n");
	}

	g->ops.mm.cache.l2_flush = stub_mm_l2_flush;
	g->ops.gr.init.fe_pwr_mode_force_on = stub_gr_init_fe_pwr_mode;

	/* Reset golden image ready bit */
	g->gr->golden_image->ready = false;
	g->gr->golden_image->size = 0x800;

	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	if (err == 0) {
		unit_err(m, "setup alloc ctrl_ctxsw failed\n");
	}

	test_gr_setup_free_obj_ctx(m, g, args);

	return (err != 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

static int gr_setup_alloc_fail_l2_flush(struct unit_module *m, struct gk20a *g)
{
	int err;

	g->allow_all = true;
	g->ops.mm.cache.l2_flush =
		gr_setup_gops.l2_flush;
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	if (err != 0) {
		unit_return_fail(m, "setup alloc l2 flush failed\n");
	}

	/* Subctx already created - redo for branch coverage */
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	if (err != 0) {
		unit_return_fail(m, "setup alloc l2 flush failed\n");
	}

	g->ops.mm.cache.l2_flush = stub_mm_l2_flush;

	return (err == 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

static int gr_setup_alloc_no_tsg_subcontext(struct unit_module *m, struct gk20a *g)
{
	int err;

	nvgpu_set_enabled(g, NVGPU_SUPPORT_TSG_SUBCONTEXTS, false);
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	nvgpu_set_enabled(g, NVGPU_SUPPORT_TSG_SUBCONTEXTS, true);
	if (err != 0) {
		unit_return_fail(m, "setup alloc disable subcontext failed\n");
	}

	return (err == 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

static void gr_setup_fake_free_obj_ctx(struct unit_module *m, struct gk20a *g)
{
	struct nvgpu_tsg_subctx *gr_subctx = gr_setup_ch->subctx;

	/* pass NULL variable*/
	gr_setup_ch->subctx = NULL;
	g->ops.gr.setup.free_subctx(gr_setup_ch);

	nvgpu_set_enabled(g, NVGPU_SUPPORT_TSG_SUBCONTEXTS, false);
	g->ops.gr.setup.free_subctx(gr_setup_ch);
	nvgpu_set_enabled(g, NVGPU_SUPPORT_TSG_SUBCONTEXTS, true);

	g->ops.gr.setup.free_gr_ctx(g, NULL);
	gr_setup_ch->subctx = gr_subctx;
}

int test_gr_setup_alloc_obj_ctx_error_injections(struct unit_module *m,
						 struct gk20a *g, void *args)
{
	int err;

	err = gr_test_setup_allocate_ch_tsg(m, g);
	if (err != 0) {
		unit_return_fail(m, "alloc setup channel failed\n");
	}

	err = gr_setup_fail_alloc(m, g);
	if (err != 0) {
		unit_return_fail(m, "setup alloc TEST-1 failed\n");
	}

	/* TEST-2 fail subctx alloc */
	err = gr_setup_fail_subctx_alloc(g);
	if (err != 0) {
		unit_return_fail(m, "setup alloc TEST-2 failed\n");
	}

	/* TEST-3 reset goldenimage size */
	err = gr_setup_alloc_fail_golden_size(m, g);
	if (err != 0) {
		unit_return_fail(m, "setup alloc TEST-3 failed\n");
	}

	/* TEST-4 fail fe_pwr_mode_on */
	err = gr_setup_alloc_fail_fe_pwr_mode(m, g);
	if (err != 0) {
		unit_return_fail(m, "setup alloc TEST-4 failed\n");
	}

	g->gr->golden_image->size = 0x800;
	gr_setup_stub_valid_ops(g);

	/* TEST-5 fail l2 flush */
	err = gr_setup_alloc_fail_l2_flush(m, g);
	if (err != 0) {
		unit_return_fail(m, "setup alloc TEST-5 failed\n");
	}

	/* TEST-6 Fake ctx free */
	gr_setup_fake_free_obj_ctx(m, g);

	/* TEST-7 Disable tsg sub-contexts */
	err = gr_setup_alloc_no_tsg_subcontext(m, g);
	if (err != 0) {
		unit_return_fail(m, "setup alloc TEST-7 failed\n");
	}

	test_gr_setup_free_obj_ctx(m, g, args);
	g->allow_all = false;

	/* TEST-8 fail ctrl_ctxsw */
	err = gr_setup_alloc_fail_ctrl_ctxsw(m, g, args);
	if (err != 0) {
		unit_return_fail(m, "setup alloc TEST-8 failed\n");
	}

	return UNIT_SUCCESS;
}

int test_gr_setup_set_preemption_mode(struct unit_module *m,
				      struct gk20a *g, void *args)
{
	int err;
	u32 compute_mode;
	u32 graphic_mode = 0;

	if (gr_setup_ch == NULL) {
		unit_return_fail(m, "failed setup with valid channel\n");
	}

	g->ops.gr.init.get_default_preemption_modes(&graphic_mode,
						&compute_mode);

	g->ops.gr.init.get_supported__preemption_modes(&graphic_mode,
						&compute_mode);

	err = g->ops.gr.setup.set_preemption_mode(gr_setup_ch, 0,
		(compute_mode & NVGPU_PREEMPTION_MODE_COMPUTE_WFI), 0);
	if (err != 0) {
		unit_return_fail(m, "setup preemption_mode failed\n");
	}

	return UNIT_SUCCESS;
}

int test_gr_setup_free_obj_ctx(struct unit_module *m,
			       struct gk20a *g, void *args)
{
	int err = 0;

	/* Restore valid ops for negative tests */
	gr_setup_restore_valid_ops(g);

	err = gr_test_setup_unbind_tsg(m, g);

	gr_test_setup_cleanup_ch_tsg(m, g);

	return (err == 0) ? UNIT_SUCCESS: UNIT_FAIL;
}

int test_gr_setup_alloc_obj_ctx(struct unit_module *m,
				struct gk20a *g, void *args)
{
	u32 tsgid = getpid();
	int err;
	bool golden_image_status;
	u32 curr_tsgid = 0;
	struct nvgpu_fifo *f = &g->fifo;

	nvgpu_posix_io_writel_reg_space(g, gr_fecs_current_ctx_r(),
							tsgid);

	g->ops.runlist.update = stub_runlist_update;

	/* Save valid gops */
	gr_setup_save_valid_ops(g);

	/* Disable those function which need register update in timeout loop */
	gr_setup_stub_valid_ops(g);

	if (f != NULL) {
		f->g = g;
	}

	/* Set a default size for golden image */
	g->gr->golden_image->size = 0x800;

	err = nvgpu_gr_global_ctx_alloc_local_golden_image(g,
			&g->gr->golden_image->local_golden_image, 0x800);
	if (err != 0) {
		unit_return_fail(m, "local golden image alloc failed\n");
	}
	err = nvgpu_gr_global_ctx_alloc_local_golden_image(g,
			&g->gr->golden_image->local_golden_image_copy, 0x800);
	if (err != 0) {
		unit_return_fail(m, "local golden image copy alloc failed\n");
	}

	/* Test with channel and tsg */
	err = gr_test_setup_allocate_ch_tsg(m, g);
	if (err != 0) {
		unit_return_fail(m, "setup channel allocation failed\n");
	}

	/* BVEC tests for variable class_num */
	gr_setup_save_class_ops(g);
	gr_setup_stub_class_ops(g);

	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, CLASS_MIN_VALUE, 0);
	if (err != 0) {
		unit_return_fail(m,
			"alloc_obj_ctx BVEC class_num min_value failed.\n");
	}

	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, CLASS_MAX_VALUE, 0);
	if (err != 0) {
		unit_return_fail(m,
			"alloc_obj_ctx BVEC class_num max_value failed.\n");
	}

	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, CLASS_VALID_VALUE, 0);
	if (err != 0) {
		unit_return_fail(m,
			"alloc_obj_ctx BVEC class_num valid_value failed.\n");
	}

	gr_setup_restore_class_ops(g);

	/* BVEC tests for variable flags */
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_DMA_COPY_A,
						FLAGS_MIN_VALUE);
	if (err != 0) {
		unit_return_fail(m,
			"alloc_obj_ctx BVEC flags min_value failed.\n");
	}

	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_DMA_COPY_A,
						FLAGS_MAX_VALUE);
	if (err != 0) {
		unit_return_fail(m,
			"alloc_obj_ctx BVEC flags max_value failed.\n");
	}

	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_DMA_COPY_A,
						FLAGS_VALID_VALUE);
	if (err != 0) {
		unit_return_fail(m,
			"alloc_obj_ctx BVEC flags valid_value failed.\n");
	}
	/* End BVEC tests */

	/* DMA_COPY should pass, but it own't allocate obj ctx */
	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_DMA_COPY_A, 0);
	if (err != 0) {
		unit_return_fail(m, "setup alloc obj_ctx failed\n");
	}

	err = g->ops.gr.setup.alloc_obj_ctx(gr_setup_ch, VOLTA_COMPUTE_A, 0);
	if (err != 0) {
		unit_return_fail(m, "setup alloc obj_ctx failed\n");
	}

	golden_image_status =
		nvgpu_gr_obj_ctx_is_golden_image_ready(g->gr->golden_image);
	if (!golden_image_status) {
		unit_return_fail(m, "No valid golden image created\n");
	}

	curr_tsgid = nvgpu_gr_ctx_get_tsgid(gr_setup_tsg->gr_ctx);
	if (curr_tsgid != gr_setup_ch->tsgid) {
		unit_return_fail(m, "Invalid tsg id\n");
	}

	return UNIT_SUCCESS;
}

struct unit_module_test nvgpu_gr_setup_tests[] = {
	UNIT_TEST(gr_setup_setup, test_gr_init_setup_ready, NULL, 0),
	UNIT_TEST(gr_setup_alloc_obj_ctx, test_gr_setup_alloc_obj_ctx, NULL, 0),
	UNIT_TEST(gr_setup_subctx_gr_ctx_buffers,
			test_gr_validate_subctx_gr_ctx_buffers, NULL, 0),
	UNIT_TEST(gr_setup_set_preemption_mode,
			test_gr_setup_set_preemption_mode, NULL, 0),
	UNIT_TEST(gr_setup_preemption_mode_errors,
			test_gr_setup_preemption_mode_errors, NULL, 2),
	UNIT_TEST(gr_setup_free_obj_ctx, test_gr_setup_free_obj_ctx, NULL, 0),
	UNIT_TEST(gr_setup_alloc_obj_ctx_error_injections,
			test_gr_setup_alloc_obj_ctx_error_injections, NULL, 2),
	UNIT_TEST(gr_setup_cleanup, test_gr_init_setup_cleanup, NULL, 0),
};

UNIT_MODULE(nvgpu_gr_setup, nvgpu_gr_setup_tests, UNIT_PRIO_NVGPU_TEST);

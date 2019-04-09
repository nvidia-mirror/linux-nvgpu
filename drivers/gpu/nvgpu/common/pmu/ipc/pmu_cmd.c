/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <nvgpu/enabled.h>
#include <nvgpu/pmu.h>
#include <nvgpu/log.h>
#include <nvgpu/timers.h>
#include <nvgpu/bug.h>
#include <nvgpu/pmu/pmuif/nvgpu_cmdif.h>
#include <nvgpu/falcon.h>
#include <nvgpu/engine_fb_queue.h>
#include <nvgpu/gk20a.h>
#include <nvgpu/string.h>
#include <nvgpu/pmu/seq.h>
#include <nvgpu/pmu/queue.h>
#include <nvgpu/pmu/cmd.h>
#include <nvgpu/pmu/msg.h>

static bool pmu_validate_cmd(struct nvgpu_pmu *pmu, struct pmu_cmd *cmd,
			struct pmu_payload *payload, u32 queue_id)
{
	struct gk20a *g = gk20a_from_pmu(pmu);
	u32 queue_size;
	u32 in_size, out_size;

	if (!PMU_IS_SW_COMMAND_QUEUE(queue_id)) {
		goto invalid_cmd;
	}


	if (cmd->hdr.size < PMU_CMD_HDR_SIZE) {
		goto invalid_cmd;
	}

	queue_size = nvgpu_pmu_queue_get_size(&pmu->queues, queue_id);

	if (cmd->hdr.size > (queue_size >> 1)) {
		goto invalid_cmd;
	}

	if (!PMU_UNIT_ID_IS_VALID(cmd->hdr.unit_id)) {
		goto invalid_cmd;
	}

	if (payload == NULL) {
		return true;
	}

	if (payload->in.buf == NULL && payload->out.buf == NULL &&
		payload->rpc.prpc == NULL) {
		goto invalid_cmd;
	}

	if ((payload->in.buf != NULL && payload->in.size == 0U) ||
	    (payload->out.buf != NULL && payload->out.size == 0U) ||
		(payload->rpc.prpc != NULL && payload->rpc.size_rpc == 0U)) {
		goto invalid_cmd;
	}

	in_size = PMU_CMD_HDR_SIZE;
	if (payload->in.buf != NULL) {
		in_size += payload->in.offset;
		in_size += g->ops.pmu_ver.get_pmu_allocation_struct_size(pmu);
	}

	out_size = PMU_CMD_HDR_SIZE;
	if (payload->out.buf != NULL) {
		out_size += payload->out.offset;
		out_size += g->ops.pmu_ver.get_pmu_allocation_struct_size(pmu);
	}

	if (in_size > cmd->hdr.size || out_size > cmd->hdr.size) {
		goto invalid_cmd;
	}


	if ((payload->in.offset != 0U && payload->in.buf == NULL) ||
	    (payload->out.offset != 0U && payload->out.buf == NULL)) {
		goto invalid_cmd;
	}

	return true;

invalid_cmd:
	nvgpu_err(g, "invalid pmu cmd :\n"
		"queue_id=%d,\n"
		"cmd_size=%d, cmd_unit_id=%d,\n"
		"payload in=%p, in_size=%d, in_offset=%d,\n"
		"payload out=%p, out_size=%d, out_offset=%d",
		queue_id, cmd->hdr.size, cmd->hdr.unit_id,
		&payload->in, payload->in.size, payload->in.offset,
		&payload->out, payload->out.size, payload->out.offset);

	return false;
}

static int pmu_write_cmd(struct nvgpu_pmu *pmu, struct pmu_cmd *cmd,
			u32 queue_id)
{
	struct gk20a *g = gk20a_from_pmu(pmu);
	struct nvgpu_timeout timeout;
	int err;

	nvgpu_log_fn(g, " ");

	nvgpu_timeout_init(g, &timeout, U32_MAX, NVGPU_TIMER_CPU_TIMER);

	do {
		err = nvgpu_pmu_queue_push(&pmu->queues, &pmu->flcn,
					   queue_id, cmd);
		if (err == -EAGAIN && nvgpu_timeout_expired(&timeout) == 0) {
			nvgpu_usleep_range(1000, 2000);
		} else {
			break;
		}
	} while (true);

	if (err != 0) {
		nvgpu_err(g, "fail to write cmd to queue %d", queue_id);
	} else {
		nvgpu_log_fn(g, "done");
	}

	return err;
}

static int pmu_payload_allocate(struct gk20a *g, struct pmu_sequence *seq,
	struct falcon_payload_alloc *alloc)
{
	struct nvgpu_pmu *pmu = &g->pmu;
	u16 buffer_size;
	int err = 0;
	u64 tmp;

	if (alloc->fb_surface == NULL &&
		alloc->fb_size != 0x0U) {

		alloc->fb_surface = nvgpu_kzalloc(g, sizeof(struct nvgpu_mem));
		if (alloc->fb_surface == NULL) {
			err = -ENOMEM;
			goto clean_up;
		}
		nvgpu_pmu_vidmem_surface_alloc(g, alloc->fb_surface,
					       alloc->fb_size);
	}

	if (nvgpu_pmu_fb_queue_enabled(&pmu->queues)) {
		buffer_size = nvgpu_pmu_seq_get_buffer_size(seq);
		nvgpu_pmu_seq_set_fbq_out_offset(seq, buffer_size);
		/* Save target address in FBQ work buffer. */
		alloc->dmem_offset = buffer_size;
		buffer_size += alloc->dmem_size;
		nvgpu_pmu_seq_set_buffer_size(seq, buffer_size);
	} else {
		tmp = nvgpu_alloc(&pmu->dmem, alloc->dmem_size);
		nvgpu_assert(tmp <= U32_MAX);
		alloc->dmem_offset = (u32)tmp;
		if (alloc->dmem_offset == 0U) {
			err = -ENOMEM;
			goto clean_up;
		}
	}

clean_up:
	return err;
}

static int pmu_cmd_payload_setup_rpc(struct gk20a *g, struct pmu_cmd *cmd,
	struct pmu_payload *payload, struct pmu_sequence *seq)
{
	struct nvgpu_pmu *pmu = &g->pmu;
	struct pmu_v *pv = &g->ops.pmu_ver;
	struct nvgpu_engine_fb_queue *queue = nvgpu_pmu_seq_get_cmd_queue(seq);
	struct falcon_payload_alloc alloc;
	int err = 0;

	nvgpu_log_fn(g, " ");

	memset(&alloc, 0, sizeof(struct falcon_payload_alloc));

	alloc.dmem_size = payload->rpc.size_rpc +
		payload->rpc.size_scratch;

	err = pmu_payload_allocate(g, seq, &alloc);
	if (err != 0) {
		goto clean_up;
	}

	alloc.dmem_size = payload->rpc.size_rpc;

	if (nvgpu_pmu_fb_queue_enabled(&pmu->queues)) {
		/* copy payload to FBQ work buffer */
		nvgpu_memcpy((u8 *)
			nvgpu_engine_fb_queue_get_work_buffer(queue) +
			alloc.dmem_offset,
			(u8 *)payload->rpc.prpc, payload->rpc.size_rpc);

		alloc.dmem_offset += nvgpu_pmu_seq_get_fbq_heap_offset(seq);

		nvgpu_pmu_seq_set_in_payload_fb_queue(seq, true);
		nvgpu_pmu_seq_set_out_payload_fb_queue(seq, true);
	} else {
		nvgpu_falcon_copy_to_dmem(&pmu->flcn, alloc.dmem_offset,
			payload->rpc.prpc, payload->rpc.size_rpc, 0);
	}

	cmd->cmd.rpc.rpc_dmem_size = payload->rpc.size_rpc;
	cmd->cmd.rpc.rpc_dmem_ptr  = alloc.dmem_offset;

	nvgpu_pmu_seq_set_out_payload(seq, payload->rpc.prpc);
	pv->pmu_allocation_set_dmem_size(pmu,
		pv->get_pmu_seq_out_a_ptr(seq),
		payload->rpc.size_rpc);
	pv->pmu_allocation_set_dmem_offset(pmu,
		pv->get_pmu_seq_out_a_ptr(seq),
		alloc.dmem_offset);

clean_up:
	if (err != 0) {
		nvgpu_log_fn(g, "fail");
	} else {
		nvgpu_log_fn(g, "done");
	}

	return err;
}

static int pmu_cmd_payload_setup(struct gk20a *g, struct pmu_cmd *cmd,
	struct pmu_payload *payload, struct pmu_sequence *seq)
{
	struct nvgpu_engine_fb_queue *fb_queue =
				nvgpu_pmu_seq_get_cmd_queue(seq);
	struct pmu_v *pv = &g->ops.pmu_ver;
	struct falcon_payload_alloc alloc;
	struct nvgpu_pmu *pmu = &g->pmu;
	void *in = NULL, *out = NULL;
	int err = 0;
	u32 offset;

	nvgpu_log_fn(g, " ");

	if (payload != NULL) {
		nvgpu_pmu_seq_set_out_payload(seq, payload->out.buf);
	}

	memset(&alloc, 0, sizeof(struct falcon_payload_alloc));

	if (payload != NULL && payload->in.offset != 0U) {
		pv->set_pmu_allocation_ptr(pmu, &in,
		((u8 *)&cmd->cmd + payload->in.offset));

		if (payload->in.buf != payload->out.buf) {
			pv->pmu_allocation_set_dmem_size(pmu, in,
			(u16)payload->in.size);
		} else {
			pv->pmu_allocation_set_dmem_size(pmu, in,
			(u16)max(payload->in.size, payload->out.size));
		}

		alloc.dmem_size = pv->pmu_allocation_get_dmem_size(pmu, in);

		if (payload->in.fb_size != 0x0U) {
			alloc.fb_size = payload->in.fb_size;
		}

		err = pmu_payload_allocate(g, seq, &alloc);
		if (err != 0) {
			goto clean_up;
		}

		*(pv->pmu_allocation_get_dmem_offset_addr(pmu, in)) =
			alloc.dmem_offset;

		if (payload->in.fb_size != 0x0U) {
			nvgpu_pmu_seq_set_in_mem(seq, alloc.fb_surface);
			nvgpu_pmu_surface_describe(g, alloc.fb_surface,
				(struct flcn_mem_desc_v0 *)
				pv->pmu_allocation_get_fb_addr(pmu, in));

			nvgpu_mem_wr_n(g, alloc.fb_surface, 0,
				payload->in.buf, payload->in.fb_size);

			if (nvgpu_pmu_fb_queue_enabled(&pmu->queues)) {
				alloc.dmem_offset +=
					nvgpu_pmu_seq_get_fbq_heap_offset(seq);
				*(pv->pmu_allocation_get_dmem_offset_addr(pmu,
									  in)) =
					alloc.dmem_offset;
			}
		} else {
			if (nvgpu_pmu_fb_queue_enabled(&pmu->queues)) {
				/* copy payload to FBQ work buffer */
				nvgpu_memcpy((u8 *)
					nvgpu_engine_fb_queue_get_work_buffer(
							fb_queue) +
					alloc.dmem_offset,
					(u8 *)payload->in.buf,
					payload->in.size);

				alloc.dmem_offset +=
					nvgpu_pmu_seq_get_fbq_heap_offset(seq);
				*(pv->pmu_allocation_get_dmem_offset_addr(pmu,
									  in)) =
					alloc.dmem_offset;

				nvgpu_pmu_seq_set_in_payload_fb_queue(seq,
								      true);
			} else {
				offset = pv->pmu_allocation_get_dmem_offset(pmu,
						in);
				nvgpu_falcon_copy_to_dmem(&pmu->flcn,
						offset, payload->in.buf,
						payload->in.size, 0);
			}
		}
		pv->pmu_allocation_set_dmem_size(pmu,
		pv->get_pmu_seq_in_a_ptr(seq),
		pv->pmu_allocation_get_dmem_size(pmu, in));
		pv->pmu_allocation_set_dmem_offset(pmu,
		pv->get_pmu_seq_in_a_ptr(seq),
		pv->pmu_allocation_get_dmem_offset(pmu, in));
	}

	if (payload != NULL && payload->out.offset != 0U) {
		pv->set_pmu_allocation_ptr(pmu, &out,
		((u8 *)&cmd->cmd + payload->out.offset));
		pv->pmu_allocation_set_dmem_size(pmu, out,
		(u16)payload->out.size);

		if (payload->in.buf != payload->out.buf) {
			alloc.dmem_size =
				pv->pmu_allocation_get_dmem_size(pmu, out);

			if (payload->out.fb_size != 0x0U) {
				alloc.fb_size = payload->out.fb_size;
			}

			err = pmu_payload_allocate(g, seq, &alloc);
			if (err != 0) {
				goto clean_up;
			}

			*(pv->pmu_allocation_get_dmem_offset_addr(pmu, out)) =
				alloc.dmem_offset;
			nvgpu_pmu_seq_set_out_mem(seq, alloc.fb_surface);
		} else {
			WARN_ON(in == NULL);
			nvgpu_pmu_seq_set_out_mem(seq,
						nvgpu_pmu_seq_get_in_mem(seq));
			pv->pmu_allocation_set_dmem_offset(pmu, out,
			pv->pmu_allocation_get_dmem_offset(pmu, in));
		}

		if (payload->out.fb_size != 0x0U) {
			nvgpu_pmu_surface_describe(g,
				nvgpu_pmu_seq_get_out_mem(seq),
				(struct flcn_mem_desc_v0 *)
				pv->pmu_allocation_get_fb_addr(pmu,
				out));
		}

		if (nvgpu_pmu_fb_queue_enabled(&pmu->queues)) {
			if (payload->in.buf != payload->out.buf) {
				*(pv->pmu_allocation_get_dmem_offset_addr(pmu,
					out)) +=
					nvgpu_pmu_seq_get_fbq_heap_offset(seq);
			}

			nvgpu_pmu_seq_set_out_payload_fb_queue(seq, true);
		}

		pv->pmu_allocation_set_dmem_size(pmu,
		pv->get_pmu_seq_out_a_ptr(seq),
		pv->pmu_allocation_get_dmem_size(pmu, out));
		pv->pmu_allocation_set_dmem_offset(pmu,
		pv->get_pmu_seq_out_a_ptr(seq),
		pv->pmu_allocation_get_dmem_offset(pmu, out));
	}

clean_up:
	if (err != 0) {
		nvgpu_log_fn(g, "fail");
		if (in != NULL) {
			nvgpu_free(&pmu->dmem,
				pv->pmu_allocation_get_dmem_offset(pmu, in));
		}
		if (out != NULL) {
			nvgpu_free(&pmu->dmem,
				pv->pmu_allocation_get_dmem_offset(pmu, out));
		}
	} else {
		nvgpu_log_fn(g, "done");
	}

	return err;
}

static int pmu_fbq_cmd_setup(struct gk20a *g, struct pmu_cmd *cmd,
	struct nvgpu_engine_fb_queue *queue, struct pmu_payload *payload,
	struct pmu_sequence *seq)
{
	struct nvgpu_pmu *pmu = &g->pmu;
	struct nv_falcon_fbq_hdr *fbq_hdr = NULL;
	struct pmu_cmd *flcn_cmd = NULL;
	u32 fbq_size_needed = 0;
	u16 heap_offset = 0;
	u64 tmp;
	int err = 0;

	fbq_hdr = (struct nv_falcon_fbq_hdr *)
		nvgpu_engine_fb_queue_get_work_buffer(queue);

	flcn_cmd = (struct pmu_cmd *)
		(nvgpu_engine_fb_queue_get_work_buffer(queue) +
		sizeof(struct nv_falcon_fbq_hdr));

	if (cmd->cmd.rpc.cmd_type == NV_PMU_RPC_CMD_ID) {
		if (payload != NULL) {
			fbq_size_needed = (u32)payload->rpc.size_rpc +
					(u32)payload->rpc.size_scratch;
		}
	} else {
		if (payload != NULL) {
			if (payload->in.offset != 0U) {
				if (payload->in.buf != payload->out.buf) {
					fbq_size_needed = payload->in.size;
				} else {
					fbq_size_needed = max(payload->in.size,
						payload->out.size);
				}
			}

			if (payload->out.offset != 0U) {
				if (payload->out.buf != payload->in.buf) {
					fbq_size_needed +=
						(u16)payload->out.size;
				}
			}
		}
	}

	tmp = fbq_size_needed +
		sizeof(struct nv_falcon_fbq_hdr) +
		cmd->hdr.size;
	nvgpu_assert(tmp <= (size_t)U32_MAX);
	fbq_size_needed = (u32)tmp;

	fbq_size_needed = ALIGN_UP(fbq_size_needed, 4);

	tmp = nvgpu_alloc(&pmu->dmem, fbq_size_needed);
	nvgpu_assert(tmp <= U32_MAX);
	heap_offset = (u16) tmp;
	if (heap_offset == 0U) {
		err = -ENOMEM;
		goto exit;
	}

	/* clear work queue buffer */
	memset(nvgpu_engine_fb_queue_get_work_buffer(queue), 0,
		nvgpu_engine_fb_queue_get_element_size(queue));

	/* Need to save room for both FBQ hdr, and the CMD */
	tmp = sizeof(struct nv_falcon_fbq_hdr) +
		   cmd->hdr.size;
	nvgpu_assert(tmp <= (size_t)U16_MAX);
	nvgpu_pmu_seq_set_buffer_size(seq, (u16)tmp);

	/* copy cmd into the work buffer */
	nvgpu_memcpy((u8 *)flcn_cmd, (u8 *)cmd, cmd->hdr.size);

	/* Fill in FBQ hdr, and offset in seq structure */
	nvgpu_assert(fbq_size_needed < U16_MAX);
	fbq_hdr->heap_size = (u16)fbq_size_needed;
	fbq_hdr->heap_offset = heap_offset;
	nvgpu_pmu_seq_set_fbq_heap_offset(seq, heap_offset);

	/*
	 * save queue index in seq structure
	 * so can free queue element when response is received
	 */
	nvgpu_pmu_seq_set_fbq_element_index(seq,
				nvgpu_engine_fb_queue_get_position(queue));

exit:
	return err;
}

int nvgpu_pmu_cmd_post(struct gk20a *g, struct pmu_cmd *cmd,
		struct pmu_payload *payload,
		u32 queue_id, pmu_callback callback, void *cb_param)
{
	struct nvgpu_pmu *pmu = &g->pmu;
	struct pmu_sequence *seq = NULL;
	struct nvgpu_engine_fb_queue *fb_queue = NULL;
	int err;

	nvgpu_log_fn(g, " ");

	if (cmd == NULL || !pmu->pmu_ready) {
		if (cmd == NULL) {
			nvgpu_warn(g, "%s(): PMU cmd buffer is NULL", __func__);
		} else {
			nvgpu_warn(g, "%s(): PMU is not ready", __func__);
		}

		WARN_ON(true);
		return -EINVAL;
	}

	if (!pmu_validate_cmd(pmu, cmd, payload, queue_id)) {
		return -EINVAL;
	}

	err = nvgpu_pmu_seq_acquire(g, &pmu->sequences, &seq, callback,
				    cb_param);
	if (err != 0) {
		return err;
	}

	cmd->hdr.seq_id = nvgpu_pmu_seq_get_id(seq);

	cmd->hdr.ctrl_flags = 0;
	cmd->hdr.ctrl_flags |= PMU_CMD_FLAGS_STATUS;
	cmd->hdr.ctrl_flags |= PMU_CMD_FLAGS_INTR;

	if (nvgpu_pmu_fb_queue_enabled(&pmu->queues)) {
		fb_queue = nvgpu_pmu_fb_queue(&pmu->queues, queue_id);
		/* Save the queue in the seq structure. */
		nvgpu_pmu_seq_set_cmd_queue(seq, fb_queue);

		/* Lock the FBQ work buffer */
		nvgpu_engine_fb_queue_lock_work_buffer(fb_queue);

		/* Create FBQ work buffer & copy cmd to FBQ work buffer */
		err = pmu_fbq_cmd_setup(g, cmd, fb_queue, payload, seq);
		if (err != 0) {
			nvgpu_err(g, "FBQ cmd setup failed");
			nvgpu_pmu_seq_release(g, &pmu->sequences, seq);
			goto exit;
		}

		/*
		 * change cmd pointer to point to FBQ work
		 * buffer as cmd copied to FBQ work buffer
		 * in call pmu_fgq_cmd_setup()
		 */
		cmd = (struct pmu_cmd *)
			(nvgpu_engine_fb_queue_get_work_buffer(fb_queue) +
			sizeof(struct nv_falcon_fbq_hdr));
	}

	if (cmd->cmd.rpc.cmd_type == NV_PMU_RPC_CMD_ID) {
		err = pmu_cmd_payload_setup_rpc(g, cmd, payload, seq);
	} else {
		err = pmu_cmd_payload_setup(g, cmd, payload, seq);
	}

	if (err != 0) {
		nvgpu_err(g, "payload setup failed");
		g->ops.pmu_ver.pmu_allocation_set_dmem_size(pmu,
			g->ops.pmu_ver.get_pmu_seq_in_a_ptr(seq), 0);
		g->ops.pmu_ver.pmu_allocation_set_dmem_size(pmu,
			g->ops.pmu_ver.get_pmu_seq_out_a_ptr(seq), 0);

		nvgpu_pmu_seq_release(g, &pmu->sequences, seq);
		goto exit;
	}

	nvgpu_pmu_seq_set_state(seq, PMU_SEQ_STATE_USED);

	err = pmu_write_cmd(pmu, cmd, queue_id);
	if (err != 0) {
		nvgpu_pmu_seq_set_state(seq, PMU_SEQ_STATE_PENDING);
	}

exit:
	if (nvgpu_pmu_fb_queue_enabled(&pmu->queues)) {
		/* Unlock the FBQ work buffer */
		nvgpu_engine_fb_queue_unlock_work_buffer(fb_queue);
	}

	nvgpu_log_fn(g, "Done, err %x", err);
	return err;
}

int nvgpu_pmu_rpc_execute(struct nvgpu_pmu *pmu, struct nv_pmu_rpc_header *rpc,
	u16 size_rpc, u16 size_scratch, pmu_callback caller_cb,
	void *caller_cb_param, bool is_copy_back)
{
	struct gk20a *g = pmu->g;
	struct pmu_cmd cmd;
	struct pmu_payload payload;
	struct rpc_handler_payload *rpc_payload = NULL;
	pmu_callback callback = NULL;
	void *rpc_buff = NULL;
	int status = 0;

	if (nvgpu_can_busy(g) == 0) {
		return 0;
	}

	if (!pmu->pmu_ready) {
		nvgpu_warn(g, "PMU is not ready to process RPC");
		status = EINVAL;
		goto exit;
	}

	if (caller_cb == NULL) {
		rpc_payload = nvgpu_kzalloc(g,
			sizeof(struct rpc_handler_payload) + size_rpc);
		if (rpc_payload == NULL) {
			status = ENOMEM;
			goto exit;
		}

		rpc_payload->rpc_buff = (u8 *)rpc_payload +
			sizeof(struct rpc_handler_payload);
		rpc_payload->is_mem_free_set =
			is_copy_back ? false : true;

		/* assign default RPC handler*/
		callback = nvgpu_pmu_rpc_handler;
	} else {
		if (caller_cb_param == NULL) {
			nvgpu_err(g, "Invalid cb param addr");
			status = EINVAL;
			goto exit;
		}
		rpc_payload = nvgpu_kzalloc(g,
			sizeof(struct rpc_handler_payload));
		if (rpc_payload == NULL) {
			status = ENOMEM;
			goto exit;
		}
		rpc_payload->rpc_buff = caller_cb_param;
		rpc_payload->is_mem_free_set = true;
		callback = caller_cb;
		WARN_ON(is_copy_back);
	}

	rpc_buff = rpc_payload->rpc_buff;
	(void) memset(&cmd, 0, sizeof(struct pmu_cmd));
	(void) memset(&payload, 0, sizeof(struct pmu_payload));

	cmd.hdr.unit_id = rpc->unit_id;
	cmd.hdr.size = (u8)(PMU_CMD_HDR_SIZE + sizeof(struct nv_pmu_rpc_cmd));
	cmd.cmd.rpc.cmd_type = NV_PMU_RPC_CMD_ID;
	cmd.cmd.rpc.flags = rpc->flags;

	nvgpu_memcpy((u8 *)rpc_buff, (u8 *)rpc, size_rpc);
	payload.rpc.prpc = rpc_buff;
	payload.rpc.size_rpc = size_rpc;
	payload.rpc.size_scratch = size_scratch;

	status = nvgpu_pmu_cmd_post(g, &cmd, &payload,
			PMU_COMMAND_QUEUE_LPQ, callback,
			rpc_payload);
	if (status != 0) {
		nvgpu_err(g, "Failed to execute RPC status=0x%x, func=0x%x",
				status, rpc->function);
		goto exit;
	}

	/*
	 * Option act like blocking call, which waits till RPC request
	 * executes on PMU & copy back processed data to rpc_buff
	 * to read data back in nvgpu
	 */
	if (is_copy_back) {
		/* wait till RPC execute in PMU & ACK */
		pmu_wait_message_cond(pmu, nvgpu_get_poll_timeout(g),
			&rpc_payload->complete, 1);
		/* copy back data to caller */
		nvgpu_memcpy((u8 *)rpc, (u8 *)rpc_buff, size_rpc);
		/* free allocated memory */
		nvgpu_kfree(g, rpc_payload);
	}

exit:
	if (status != 0) {
		if (rpc_payload != NULL) {
			nvgpu_kfree(g, rpc_payload);
		}
	}

	return status;
}

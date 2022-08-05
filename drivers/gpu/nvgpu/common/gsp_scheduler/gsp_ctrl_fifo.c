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
#include <nvgpu/log.h>
#include <nvgpu/gsp.h>
#include <nvgpu/gsp_sched.h>
#include <nvgpu/nvs.h>

#include "gsp_runlist.h"
#include "gsp_scheduler.h"
#include "gsp_ctrl_fifo.h"
#include "ipc/gsp_cmd.h"

#ifdef CONFIG_NVS_PRESENT
static int gsp_ctrl_fifo_get_queue_info(struct gk20a *g,
        struct nvgpu_gsp_ctrl_fifo_info *ctrl_fifo, enum queue_type qtype)
{
    int err = 0;
    u8 mask;
    enum nvgpu_nvs_ctrl_queue_num queue_num;
    enum nvgpu_nvs_ctrl_queue_direction queue_direction;
    struct nvgpu_nvs_ctrl_queue *queue;

    nvgpu_gsp_dbg(g, " ");

    switch (qtype) {
        case CONTROL_QUEUE:
            mask = NVGPU_NVS_CTRL_FIFO_QUEUE_EXCLUSIVE_CLIENT_WRITE;
            queue_num = NVGPU_NVS_NUM_CONTROL;
            queue_direction = NVGPU_NVS_DIR_CLIENT_TO_SCHEDULER;
            break;
        case RESPONSE_QUEUE:
            mask = NVGPU_NVS_CTRL_FIFO_QUEUE_EXCLUSIVE_CLIENT_READ;
            queue_num = NVGPU_NVS_NUM_CONTROL;
            queue_direction = NVGPU_NVS_DIR_SCHEDULER_TO_CLIENT;
            break;
        default:
            nvgpu_err(g, "queue type invalid");
            err = -EINVAL;
            goto exit;
    }

    /* below functions will be removed/changed once UMD support is there. */
    queue = nvgpu_nvs_ctrl_fifo_get_queue(g->sched_ctrl_fifo, queue_num,
            queue_direction, &mask);
    if (queue == NULL) {
        nvgpu_err(g, "queue allocation failed");
        err = -EFAULT;
        goto exit;
    }
    /* below functions will be removed/changed once UMD support is there. */
    err = nvgpu_nvs_buffer_alloc(g->sched_ctrl_fifo, NVS_QUEUE_DEFAULT_SIZE,
            mask, queue);
    if (err != 0) {
        nvgpu_err(g, "gsp buffer allocation failed");
        goto exit;
    }
    ctrl_fifo->fifo_addr_lo = u64_lo32(queue->mem.gpu_va);
    ctrl_fifo->fifo_addr_hi = u64_hi32(queue->mem.gpu_va);
    ctrl_fifo->queue_size = GSP_CTRL_FIFO_QUEUE_SIZE;
    ctrl_fifo->queue_entries = GSP_CTRL_FIFO_QUEUE_ENTRIES;
    ctrl_fifo->qtype  = qtype;

exit:
    return err;

}

/* get and send the control fifo info to gsp */
int nvgpu_gsp_sched_send_queue_info(struct gk20a *g, enum queue_type qtype)
{
    int err = 0;
    struct nv_flcn_cmd_gsp cmd = { };
    struct nvgpu_gsp_ctrl_fifo_info ctrl_fifo = {};

    nvgpu_gsp_dbg(g, " ");

    /* below function will be removed/changed once UMD support is there. */
    err = gsp_ctrl_fifo_get_queue_info(g, &ctrl_fifo, qtype);
    if (err != 0) {
        nvgpu_err(g, "getting fifo queue info failed");
        goto exit;
    }

    cmd.cmd.ctrl_fifo.fifo_addr_lo = ctrl_fifo.fifo_addr_lo;
    cmd.cmd.ctrl_fifo.fifo_addr_hi = ctrl_fifo.fifo_addr_hi;
    cmd.cmd.ctrl_fifo.queue_size = ctrl_fifo.queue_size;
    cmd.cmd.ctrl_fifo.qtype = ctrl_fifo.qtype;
    cmd.cmd.ctrl_fifo.queue_entries = ctrl_fifo.queue_entries;

    err = gsp_send_cmd_and_wait_for_ack(g, &cmd, NV_GSP_UNIT_CONTROL_INFO_SEND,
            sizeof(struct nvgpu_gsp_ctrl_fifo_info));
    if (err != 0) {
        nvgpu_err(g, "sending control fifo queue to GSP failed");
    }

exit:
    return err;
}
#endif /* CONFIG_NVS_PRESENT*/
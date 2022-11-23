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

#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include <unit/io.h>
#include <unit/unit.h>
#include <unit/utils.h>

#include <nvgpu/channel.h>
#include <nvgpu/error_notifier.h>
#include <nvgpu/tsg.h>
#include <nvgpu/gk20a.h>
#include <nvgpu/fifo/userd.h>
#include <nvgpu/runlist.h>
#include <nvgpu/fuse.h>
#include <nvgpu/dma.h>
#include <nvgpu/gr/ctx.h>
#include <nvgpu/types.h>

#include "common/gr/ctx_priv.h"
#include <nvgpu/posix/posix-fault-injection.h>
#include <nvgpu/posix/posix-channel.h>

#include "hal/fifo/tsg_gk20a.h"

#include "hal/init/hal_gv11b.h"

#include "../fifo/nvgpu-fifo-common.h"
#include <nvgpu/nvs.h>
#include <nvgpu/nvs-control-interface-parser.h>
#include "nvgpu-nvs.h"

#ifdef CONFIG_KMD_SCHEDULING_WORKER_THREAD
#define NUM_CHANNELS     5
#define NUM_DOMAINS      4
#define NUM_TSGS         NUM_CHANNELS
#define NUM_TESTS        25

const char *dom_names[] = {"dom0", "dom1", "dom2", "dom3", "dom4"};
const char *message_error_type[] = {"NVS_DOMAIN_MSG_TYPE_CTRL_ERROR",
	"NVS_DOMAIN_MSG_TYPE_CTRL_GET_CAPS_INFO",
	"NVS_DOMAIN_MSG_TYPE_CTRL_SWITCH_DOMAIN"};

struct nvgpu_nvs_context {
	struct gk20a *g;
	struct gpu_ops gops;
	struct nvgpu_mem pdb_mem;
	struct mm_gk20a mm;
	struct vm_gk20a vm;
	struct nvgpu_channel *ch[NUM_CHANNELS];
	struct nvgpu_tsg *tsg[NUM_TSGS];
	struct nvgpu_setup_bind_args bind_args;

	struct nvs_domain_ctrl_fifo_user user;
	struct nvgpu_nvs_ctrl_queue *queue1;
	struct nvgpu_nvs_ctrl_queue *queue2;
	struct nvs_control_fifo_receiver *client_receiver;
	struct nvs_control_fifo_sender *client_sender;
	struct nvgpu_nvs_domain *domain[NUM_DOMAINS + 1];

	/* Used to verify the domain switch request */
	struct nvgpu_runlist_domain *updated_domain;
};

struct nvgpu_nvs_context nvs_context = {0};

static void stub_gr_intr_flush_channel_tlb(struct gk20a *g)
{
}
static int stub_mm_l2_flush(struct gk20a *g, bool invalidate)
{
	return 0;
}

static int stub_os_channel_alloc_usermode_buffers(struct nvgpu_channel *ch,
		struct nvgpu_setup_bind_args *args)
{
	int err;
	struct gk20a *g = ch->g;

	err = nvgpu_dma_alloc(g, NVGPU_CPU_PAGE_SIZE, &ch->usermode_userd);
	if (err != 0) {
		return err;
	}

	err = nvgpu_dma_alloc(g, NVGPU_CPU_PAGE_SIZE, &ch->usermode_gpfifo);
	if (err != 0) {
		return err;
	}

	return err;
}

static int stub_runlist_update(struct gk20a *g, struct nvgpu_runlist *rl,
		struct nvgpu_channel *ch, bool add, bool wait_for_finish)
{
	return 0;
}

static void stub_runlist_hw_submit(struct gk20a *g, struct nvgpu_runlist *runlist)
{
	struct nvgpu_runlist_domain *domain = runlist->domain;
	nvs_context.updated_domain = domain;
}

static int stub_runlist_check_pending(struct gk20a *g, struct nvgpu_runlist *runlist)
{
	if (runlist->domain->domain_id == nvs_context.domain[NUM_DOMAINS]->id) {
		return 1;
	}

	return 0;
}

static int test_nvs_worker(struct unit_module *m,
		struct gk20a *g, void *args)
{
	int i = 0, err = 0;
	u32 message_type;
	u32 message_sequence_tag = 0;
	u64 preempt_grace_ms = 5000ULL;
	u64 read_wait_time = 200ULL;
	u64 start_time;
	u64 current_time;
	struct nvgpu_nvs_context *nvs_context = (struct nvgpu_nvs_context *)args;

	bool error_case = false;
	bool error_timedout = false;
	bool shadow_domain = false;
	u64 domain_switch_id = 0;
	struct nvgpu_nvs_domain *dom_matched = NULL;

	u32 fifo_msg_type;
	struct nvs_domain_msg_ctrl_get_caps_req *client_request_caps =
		(struct nvs_domain_msg_ctrl_get_caps_req *)nvs_context->client_sender->internal_buffer;
	struct nvs_domain_msg_ctrl_switch_domain_req *client_request_switch =
		(struct nvs_domain_msg_ctrl_switch_domain_req *)nvs_context->client_sender->internal_buffer;
	struct nvs_domain_msg_ctrl_get_caps_resp *client_received_caps_response =
		(struct nvs_domain_msg_ctrl_get_caps_resp *)nvs_context->client_receiver->internal_buffer;
	struct nvs_domain_msg_ctrl_switch_domain_resp *client_received_switch_response =
		(struct nvs_domain_msg_ctrl_switch_domain_resp *)nvs_context->client_receiver->internal_buffer;
	struct nvs_domain_msg_ctrl_error_resp *client_received_error_response =
		(struct nvs_domain_msg_ctrl_error_resp *)nvs_context->client_receiver->internal_buffer;

	for (i = 0; i < NUM_TESTS; i++) {
		message_type = rand()%3;
		error_case = rand()%2;
		if (error_case) {
			error_timedout = rand()%2;
		}
		// 0 is success, 1 is shadow_domain, 2 is error
		if (rand()%3 == 1) {
			shadow_domain = true;
		}

		if (nvs_control_fifo_sender_can_write(nvs_context->client_sender) != 0) {
			unit_err(m, "Sender Cannot Write\n");
			return UNIT_FAIL;
		}

		dom_matched = NULL;

		//Send To NVS
		if (message_type == 0) {
			fifo_msg_type = NVS_DOMAIN_MSG_TYPE_CTRL_GET_CAPS_INFO;
			client_request_caps->client_version_major = NVS_DOMAIN_SCHED_VERSION_MAJOR;
			client_request_caps->client_version_minor = NVS_DOMAIN_SCHED_VERSION_MINOR;
			client_request_caps->client_version_patch = NVS_DOMAIN_SCHED_VERSION_PATCH;
			if (error_case) {
				client_request_caps->client_version_patch = 127U;
			}
		} else if (message_type == 1) {
			fifo_msg_type = NVS_DOMAIN_MSG_TYPE_CTRL_SWITCH_DOMAIN;
			if (error_case) {
				if (error_timedout) {
					domain_switch_id = nvs_context->domain[NUM_DOMAINS]->id;
				} else {
					domain_switch_id = rand()%(U64_MAX - NUM_DOMAINS) + NUM_DOMAINS + 1;
				}
			} else if (shadow_domain) {
				domain_switch_id = NVS_DOMAIN_CTRL_DOMAIN_ID_ALL;
			} else {
				domain_switch_id = nvs_context->domain[rand()%NUM_DOMAINS]->id;
			}
			unit_info(m, "Request: [Switch Domain]: Id: %llu, error_case: %d\n", domain_switch_id, error_case);
			client_request_switch->domain_id = domain_switch_id;
		} else {
			fifo_msg_type = rand()%(U32_MAX - 2);
			if (fifo_msg_type > 0) {
				fifo_msg_type += 2;
			}
			client_request_switch->domain_id = domain_switch_id;
		}

		nvs_control_fifo_sender_write_message(nvs_context->client_sender,
			fifo_msg_type, message_sequence_tag,
				nvgpu_safe_cast_s64_to_u64(nvgpu_current_time_ns()));

		start_time = nvgpu_safe_cast_s64_to_u64(nvgpu_current_time_ms());
		current_time = start_time;

		do {
			nvgpu_msleep(read_wait_time);
			err = nvs_control_fifo_receiver_can_read(nvs_context->client_receiver);
			current_time = nvgpu_safe_cast_s64_to_u64(nvgpu_current_time_ms());
			if (err == 0) {
				break;
			}
			read_wait_time = nvgpu_safe_mult_u64(read_wait_time, 2ULL);
		} while (nvgpu_safe_sub_u64(current_time, start_time) <= preempt_grace_ms);

		if (err != 0) {
			unit_info(m, "Ring buffer communication Timed out: Message Type: %u, Message Sequence: %u, Timestamp %llu\n",
				 fifo_msg_type, nvs_context->client_receiver->msg_sequence, nvs_context->client_receiver->msg_timestamp_ns);
			return UNIT_FAIL;
		}

		nvs_control_fifo_read_message(nvs_context->client_receiver);
		if (nvs_context->client_receiver->msg_sequence == message_sequence_tag) {
			if (fifo_msg_type == NVS_DOMAIN_MSG_TYPE_CTRL_GET_CAPS_INFO) {
				if (error_case) {
					unit_assert(client_received_caps_response->client_version_status ==
						NVS_DOMAIN_MSG_CTRL_GET_CAPS_RESP_CLIENT_VERSION_STATUS_FAILED, goto fail);
				} else {
					unit_assert(client_received_caps_response->client_version_status ==
						NVS_DOMAIN_MSG_CTRL_GET_CAPS_RESP_CLIENT_VERSION_STATUS_OK, goto fail);
				}
			} else if (fifo_msg_type == NVS_DOMAIN_MSG_TYPE_CTRL_SWITCH_DOMAIN) {
				if (error_case) {
					unit_assert(client_received_switch_response->status !=
						NVS_DOMAIN_MSG_TYPE_CTRL_SWITCH_DOMAIN_STATUS_SUCCESS, goto fail);
				} else {
					unit_assert(client_received_switch_response->status ==
						NVS_DOMAIN_MSG_TYPE_CTRL_SWITCH_DOMAIN_STATUS_SUCCESS, goto fail);
					unit_assert(nvs_context->updated_domain != NULL && nvs_context->updated_domain->domain_id
						== domain_switch_id, goto fail);
					if (domain_switch_id == NVS_DOMAIN_CTRL_DOMAIN_ID_ALL) {
						dom_matched = g->scheduler->shadow_domain;
					} else {
						dom_matched = nvgpu_nvs_domain_by_id_locked(g, domain_switch_id);
					}
				}
			} else {
				unit_assert(nvs_context->client_receiver->msg_type == NVS_DOMAIN_MSG_TYPE_CTRL_ERROR, goto fail);
				unit_assert(client_received_error_response->error_code == NVS_DOMAIN_MSG_CTRL_ERROR_UNHANDLED_MESSAGE, goto fail);
			}
			unit_info(m, "Ring buffer communication was correct: Message Type: %s, Message Sequence: %u, Timestamp: %llu\n",
				nvs_context->client_receiver->msg_type < 3 ? message_error_type[nvs_context->client_receiver->msg_type] :
					message_error_type[0],
				nvs_context->client_receiver->msg_sequence,
				nvs_context->client_receiver->msg_timestamp_ns);
			if (nvs_context->client_receiver->msg_type == NVS_DOMAIN_MSG_TYPE_CTRL_SWITCH_DOMAIN) {
				unit_info(m, "Domain Switch Requested: Domain: [%s], Response: Status: %d, Duration: %lu\n",
					dom_matched == NULL ? "Invalid domain" : nvgpu_nvs_domain_get_name(dom_matched),
					client_received_switch_response->status, client_received_switch_response->switch_ns);
			} else if (nvs_context->client_receiver->msg_type == NVS_DOMAIN_MSG_TYPE_CTRL_GET_CAPS_INFO) {
				unit_info(m, "Response: Status: %d\n", client_received_caps_response->client_version_status);
			} else {
				unit_info(m, "Invalid request: %u, Response: Status: %d\n", nvs_context->client_receiver->msg_type,
					client_received_error_response->error_code);
			}
		} else {
			unit_info(m, "Ring buffer communication was Incorrect: Message Type: %u, Message Sequence: %u, Timestamp %llu\n",
				 fifo_msg_type, nvs_context->client_receiver->msg_sequence, nvs_context->client_receiver->msg_timestamp_ns);
			return UNIT_FAIL;
		}

		shadow_domain = false;

		message_sequence_tag++;
	}

	return UNIT_SUCCESS;
fail:
	return UNIT_FAIL;
}

int test_nvs_setup_sw(struct unit_module *m,
		struct gk20a *g, void *args)
{
	int ret = 0, err = 0;
	int i = 0;
	struct nvgpu_nvs_context *nvs_context = (struct nvgpu_nvs_context *)args;
	nvs_context->gops = g->ops;
	u8 mask;

	srand(time(NULL));

	g->ops.gr.intr.flush_channel_tlb = stub_gr_intr_flush_channel_tlb;
	g->ops.mm.cache.l2_flush = stub_mm_l2_flush;	/* bug 2621189 */
	g->os_channel.alloc_usermode_buffers = stub_os_channel_alloc_usermode_buffers;
	g->ops.runlist.update = stub_runlist_update;
	g->ops.runlist.hw_submit = stub_runlist_hw_submit;
	g->ops.runlist.check_pending = stub_runlist_check_pending;

	memset(&nvs_context->mm, 0, sizeof(nvs_context->mm));
	memset(&nvs_context->vm, 0, sizeof(nvs_context->vm));
	nvs_context->mm.g = g;
	nvs_context->vm.mm = &nvs_context->mm;
	err = nvgpu_dma_alloc(g, NVGPU_CPU_PAGE_SIZE, &nvs_context->pdb_mem);
	unit_assert(err == 0, goto done);
	nvs_context->vm.pdb.mem = &nvs_context->pdb_mem;

	/* bind arguments */
	memset(&nvs_context->bind_args, 0, sizeof(nvs_context->bind_args));
	nvs_context->bind_args.num_gpfifo_entries = 32;
	nvs_context->bind_args.flags |=
		NVGPU_SETUP_BIND_FLAGS_SUPPORT_DETERMINISTIC;
	nvs_context->bind_args.flags |=
		NVGPU_SETUP_BIND_FLAGS_USERMODE_SUPPORT;

	nvgpu_set_enabled(g, NVGPU_SUPPORT_NVS, true);

	err = nvgpu_nvs_init(g);

	nvs_context->queue1 = nvgpu_nvs_ctrl_fifo_get_queue(g->sched_ctrl_fifo, NVGPU_NVS_NUM_CONTROL,
			NVGPU_NVS_DIR_CLIENT_TO_SCHEDULER, &mask);

	if (nvs_context->queue1 != NULL) {
		if (nvgpu_nvs_buffer_alloc(g->sched_ctrl_fifo, NVS_QUEUE_DEFAULT_SIZE, mask, nvs_context->queue1) != 0) {
			unit_err(m, "Sched Fifo init failed!\n");
			return UNIT_FAIL;
		} else {
			nvs_context->client_sender = nvs_control_fifo_sender_initialize(g, (struct nvs_domain_msg_fifo * const)nvs_context->queue1->mem.cpu_va,
				NVS_QUEUE_DEFAULT_SIZE);
			if (nvs_context->client_sender == NULL) {
				unit_err(m, "Sched Fifo init failed!\n");
				return UNIT_FAIL;
			}
		}
	}

	nvs_context->queue2 = nvgpu_nvs_ctrl_fifo_get_queue(g->sched_ctrl_fifo, NVGPU_NVS_NUM_CONTROL,
			NVGPU_NVS_DIR_SCHEDULER_TO_CLIENT, &mask);

	if (nvs_context->queue2 != NULL) {
		if (nvgpu_nvs_buffer_alloc(g->sched_ctrl_fifo, NVS_QUEUE_DEFAULT_SIZE, mask, nvs_context->queue2) != 0) {
			unit_err(m, "Sched Fifo init failed!\n");
			return UNIT_FAIL;
		} else {
			nvs_context->client_receiver = nvs_control_fifo_receiver_initialize(g, (struct nvs_domain_msg_fifo * const)nvs_context->queue2->mem.cpu_va,
				NVS_QUEUE_DEFAULT_SIZE);
			if (nvs_context->client_receiver == NULL) {
				unit_err(m, "Sched Fifo init failed!\n");
				return UNIT_FAIL;
			}
		}
	}

	nvgpu_init_list_node(&nvs_context->user.sched_ctrl_list);
	nvgpu_nvs_ctrl_fifo_add_user(g->sched_ctrl_fifo, &nvs_context->user);

	for (i = 0; i < NUM_DOMAINS + 1; i++) {
		if (nvgpu_nvs_add_domain(g, dom_names[i], 100, i == NUM_DOMAINS ? 100 : 0,
				&nvs_context->domain[i]) != 0) {
			unit_err(m, "Creation of nvs_context->Domain[i] failed!\n");
			return UNIT_FAIL;
		} else {
			unit_info(m, "Domain Id [%llu]\n", nvs_context->domain[i]->id);
		}
	}

	for (i = 0; i < NUM_CHANNELS; i++) {
		u64 domain_id = rand()%NUM_DOMAINS;

		nvs_context->tsg[i] = nvgpu_tsg_open(g, getpid());
		unit_assert(nvs_context->tsg[i] != NULL, goto done);

		nvgpu_nvs_domain_get(g, nvs_context->domain[domain_id]);
		nvgpu_tsg_bind_domain(nvs_context->tsg[i], nvs_context->domain[domain_id]);
		nvgpu_nvs_domain_put(g, nvs_context->domain[domain_id]);

		nvs_context->ch[i] = nvgpu_channel_open_new(g, 0,
				false, getpid(), getpid());
		unit_assert(nvs_context->ch[i] != NULL, goto done);

		err = nvgpu_tsg_bind_channel(nvs_context->tsg[i], nvs_context->ch[i]);
		unit_assert(err == 0, goto done);

		nvs_context->ch[i]->vm = &nvs_context->vm;

		err = nvgpu_channel_setup_bind(nvs_context->ch[i], &nvs_context->bind_args);
	}

	if (test_nvs_worker(m, g, args) != 0) {
		unit_err(m, "Creation of nvs_context->Domain[i] failed!\n");
		return UNIT_FAIL;
	}

done:
	return ret;
}

int test_nvs_remove_sw(struct unit_module *m,
		struct gk20a *g, void *args)
{
	int ret = 0;
	int i = 0;
	struct nvgpu_nvs_context *nvs_context = (struct nvgpu_nvs_context *)args;

	for (i = 0; i < NUM_CHANNELS; i++) {
		ret = nvgpu_tsg_unbind_channel(nvs_context->tsg[i], nvs_context->ch[i], true);
		unit_assert(ret == 0, goto done);

		nvgpu_channel_close(nvs_context->ch[i]);
		nvgpu_ref_put(&nvs_context->tsg[i]->refcount, nvgpu_tsg_release);
	}

	for (i = 0; i < NUM_DOMAINS + 1; i++) {
		if (nvs_context->domain[i] != NULL) {
			nvgpu_nvs_del_domain(g, nvs_context->domain[i]->id);
			nvs_context->domain[i] = NULL;
		}
	}

	if (nvs_context->client_sender != NULL) {
		nvs_control_fifo_sender_exit(g, nvs_context->client_sender);
		nvs_context->client_sender = NULL;
	}

	if (nvs_context->client_receiver != NULL) {
		nvs_control_fifo_receiver_exit(g, nvs_context->client_receiver);
		nvs_context->client_receiver = NULL;
	}

	if (nvs_context->queue1 != NULL)
		nvgpu_nvs_buffer_free(g->sched_ctrl_fifo, nvs_context->queue1);
	if (nvs_context->queue2 != NULL)
		nvgpu_nvs_buffer_free(g->sched_ctrl_fifo, nvs_context->queue2);

	nvgpu_nvs_ctrl_fifo_remove_user(g->sched_ctrl_fifo, &nvs_context->user);
	nvgpu_nvs_remove_support(g);

	g->ops = nvs_context->gops;

done:
	return ret;
}
#endif

struct unit_module_test nvgpu_nvs_tests[] = {
	UNIT_TEST(init_support, test_fifo_init_support, &nvs_context, 0),
#ifdef CONFIG_KMD_SCHEDULING_WORKER_THREAD
	UNIT_TEST(setup_sw, test_nvs_setup_sw, &nvs_context, 0),
	UNIT_TEST(nvs_remove_support, test_nvs_remove_sw, &nvs_context, 0),
#endif
	UNIT_TEST(remove_support, test_fifo_remove_support, NULL, 0),
};

UNIT_MODULE(nvgpu_nvs, nvgpu_nvs_tests, UNIT_PRIO_NVGPU_TEST);

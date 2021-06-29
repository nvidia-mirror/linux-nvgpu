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
#ifndef UNIT_NVGPU_RC_H
#define UNIT_NVGPU_RC_H

#include <nvgpu/types.h>

struct gk20a;
struct unit_module;

/** @addtogroup SWUTS-nvgpu-rc
 *  @{
 *
 * Software Unit Test Specification for nvgpu-rc
 */

/**
 * Test specification for: test_rc_init
 *
 * Description: Environment initialization for rc tests
 *
 * Test Type: Other (setup)
 *
 * Input: None
 *
 * Steps:
 * - init FIFO register space.
 * - init HAL parameters for gv11b.
 * - init fifo support for Channel and TSG
 * - init Runlist setup
 * - open a TSG
 * - open a new Channel
 * - allocate memory for posix_channel
 * - bind Channel to TSG
 *
 * Output: Returns PASS if all the above steps are successful. FAIL otherwise.
 */
int test_rc_init(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_deinit
 *
 * Description: Environment de-initialization for rc tests
 *
 * Test Type: Other (cleanup)
 *
 * Input: test_rc_init run for this GPU
 *
 * Steps:
 * - unbind Channel from TSG
 * - free posix_channel
 * - close Channel
 * - close TSG
 * - remove FIFO support
 * - clear FIFO register space
 *
 * Output: Returns PASS if all the above steps are successful. FAIL otherwise.
 */
int test_rc_deinit(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_fifo_recover
 *
 * Description: Coverage test for nvgpu_rc_fifo_recover
 *
 * Test Type: Feature
 *
 * Targets: nvgpu_rc_fifo_recover
 *
 * Input: test_rc_init run for this GPU
 *
 * Steps:
 * - initialize Channel error_notifier
 * - set g->sw_quiesce_pending = true
 * - invoke nvgpu_rc_fifo_recover
 *
 * Output: Cover all branch in safety build.
 */
int test_rc_fifo_recover(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_ctxsw_timeout
 *
 * Description: Coverage test for nvgpu_rc_ctxsw_timeout
 *
 * Test Type: Feature
 *
 * Targets: nvgpu_rc_ctxsw_timeout
 *
 * Input: test_rc_init run for this GPU
 *
 * Steps:
 * - initialize Channel error_notifier
 * - set g->sw_quiesce_pending = true
 * - invoke nvgpu_rc_ctxsw_timeout
 * - verfy that error_notifier is set to NVGPU_ERR_NOTIFIER_FIFO_ERROR_IDLE_TIMEOUT
 *
 * Output: Cover all branch in safety build.
 */
int test_rc_ctxsw_timeout(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_runlist_update
 *
 * Description: Coverage test for nvgpu_rc_runlist_update
 *
 * Test Type: Feature
 *
 * Targets: nvgpu_rc_runlist_update
 *
 * Input: test_rc_init run for this GPU
 *
 * Steps:
 * - initialize Channel error_notifier
 * - set g->sw_quiesce_pending = true
 * - invoke nvgpu_rc_runlist_update
 *
 * Output: Cover all branch in safety build.
 */
int test_rc_runlist_update(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_preempt_timeout
 *
 * Description: Coverage test for nvgpu_rc_preempt_timeout
 *
 * Test Type: Feature
 *
 * Targets: nvgpu_rc_preempt_timeout
 *
 * Input: test_rc_init run for this GPU
 *
 * Steps:
 * - initialize Channel error_notifier
 * - set g->sw_quiesce_pending = true
 * - invoke nvgpu_rc_preempt_timeout
 * - verfy that error_notifier is set to NVGPU_ERR_NOTIFIER_FIFO_ERROR_IDLE_TIMEOUT
 *
 * Output: Cover all branch in safety build.
 */
int test_rc_preempt_timeout(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_gr_fault
 *
 * Description: Coverage test for nvgpu_rc_gr_fault
 *
 * Test Type: Feature
 *
 * Targets: nvgpu_rc_gr_fault
 *
 * Input: test_rc_init run for this GPU
 *
 * Steps:
 * - initialize Channel error_notifier
 * - set g->sw_quiesce_pending = true
 * - invoke nvgpu_rc_gr_fault
 *
 * Output: Cover all branch in safety build.
 */
int test_rc_gr_fault(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_sched_error_bad_tsg
 *
 * Description: Coverage test for nvgpu_rc_sched_error_bad_tsg
 *
 * Test Type: Feature
 *
 * Targets: nvgpu_rc_sched_error_bad_tsg
 *
 * Input: test_rc_init run for this GPU
 *
 * Steps:
 * - initialize Channel error_notifier
 * - set g->sw_quiesce_pending = true
 * - invoke nvgpu_rc_sched_error_bad_tsg
 *
 * Output: Cover all branch in safety build.
 */
int test_rc_sched_error_bad_tsg(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_tsg_and_related_engines
 *
 * Description: Coverage test for nvgpu_rc_tsg_and_related_engines
 *
 * Test Type: Feature
 *
 * Targets: nvgpu_rc_tsg_and_related_engines
 *
 * Input: test_rc_init run for this GPU
 *
 * Steps:
 * - initialize Channel error_notifier
 * - set g->sw_quiesce_pending = true
 * - invoke nvgpu_rc_tsg_and_related_engines
 *
 * Output: Cover all branch in safety build.
 */
int test_rc_tsg_and_related_engines(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_mmu_fault
 *
 * Description: Coverage test for nvgpu_rc_mmu_fault
 *
 * Test Type: Feature
 *
 * Targets: nvgpu_rc_mmu_fault
 *
 * Input: test_rc_init run for this GPU
 *
 * Steps:
 * - initialize Channel error_notifier
 * - set g->sw_quiesce_pending = true
 * - set invalid_id
 *	- invoke nvgpu_rc_mmu_fault
 * - set id_type_tsg
 *	- invoke nvgpu_rc_mmu_fault
 * - set id_type_non_tsg
 *  - invoke nvgpu_rc_mmu_fault
 *
 * Output: Cover all branch in safety build.
 */
int test_rc_mmu_fault(struct unit_module *m, struct gk20a *g, void *args);

/**
 * Test specification for: test_rc_pbdma_fault
 *
 * Description: Coverage test for nvgpu_rc_pbdma_fault
 *
 * Test Type: Feature, Boundary Value
 *
 * Targets: nvgpu_rc_pbdma_fault
 *
 * Input: test_rc_init run for this GPU
 *
 * Equivalence classes:
 * Variable: error_notifier
 * - Valid: [NVGPU_ERR_NOTIFIER_FIFO_ERROR_IDLE_TIMEOUT, NVGPU_ERR_NOTIFIER_PBDMA_PUSHBUFFER_CRC_MISMATCH]
 * - Invalid: [NVGPU_ERR_NOTIFIER_INVAL, INT_MAX]
 * Variable: chsw state
 * - Valid: [NVGPU_PBDMA_CHSW_STATUS_INVALID, NVGPU_PBDMA_CHSW_STATUS_SWITCH]
 * - Invalid: [NVGPU_PBDMA_CHSW_STATUS_SWITCH + 1, INT_MAX]
 * Variable: id_type
 * - Valid: [PBDMA_STATUS_ID_TYPE_CHID, PBDMA_STATUS_ID_TYPE_TSGID]
 * - Invalid: [PBDMA_STATUS_ID_TYPE_TSGID + 1, PBDMA_STATUS_ID_TYPE_INVALID]
 *
 * Steps:
 * - initialize Channel error_notifier
 * - test with valid and invalid error notifier values types
 * - set g->sw_quiesce_pending = true
 * - For each branch check with the following pbdma_status values
 * - set chsw_status to chsw_valid_or_save
 *   - set id_type to TSG
 *   - set id_type to Channel
 *     - set Channel Id to Invalid
 *     - set Channel Id to a channel without TSG
 *     - set Channel Id to a channel with a valid TSG
 *     - set id_type to chid, tsgid, tsgid + 1, tsgid + 1 + random, invalid_id
 *   - verify that nvgpu_rc_pbdma_fault fails for invalid id_types and invalid channel ids and succeeds otherwise.
 * - set chsw_status to is_chsw_load_or_switch
 *   - set id_type to TSG
 *   - set id_type to Channel
 *     - set Channel Id to Invalid
 *     - set Channel Id to a channel without TSG
 *     - set Channel Id to a channel with a valid TSG
 *     - set id_type to chid, tsgid, tsgid + 1, tsgid + 1 + random, invalid_id
 *   - verify that nvgpu_rc_pbdma_fault fails for invalid id_types and invalid channel ids and succeeds otherwise.
 * - set chsw_status to chsw_invalid and verify that nvgpu_rc_pbdma_fault succeeds.
 * - set chsw_status to invalid states and verify that nvgpu_rc_pbdma_fault fails.
 *
 * Output: Returns PASS if nvgpu_rc_pbdma_fault succeeds for valid inputs
 *         and fails for invalid inputs. Returns FAIL otherwise.
 */
int test_rc_pbdma_fault(struct unit_module *m, struct gk20a *g, void *args);

/** @} */

#endif /* UNIT_NVGPU_RC_H */

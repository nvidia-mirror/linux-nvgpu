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

#include <nvgpu/nvgpu_err.h>
#include <nvgpu/nvgpu_err_info.h>
#include <nvgpu/static_analysis.h>
#include "common/cic/cic_priv.h"
#include "cic_gv11b.h"

/*
 * A flag to enable/disable hw error injection.
 */
#ifdef CONFIG_NVGPU_INJECT_HWERR
#define INJECT_TYPE	(INJECT_HW)
#else
#define INJECT_TYPE	(INJECT_SW)
#endif

/* This look-up table initializes the list of hw units and their errors.
 * It also specifies the error injection mechanism supported, for each error.
 * In case of hw error injection support, this initialization will be overriden
 * by the values provided from the hal layes of corresponding hw units.
 */
struct nvgpu_err_hw_module gv11b_err_lut[] = {
	{
		.name = "host",
		.hw_unit = (u32)NVGPU_ERR_MODULE_HOST,
		.num_instances = 1U,
		.num_errs = 17U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_CRITERR("pfifo_bind_error",
					GPU_HOST_PFIFO_BIND_ERROR, INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pfifo_sched_error",
					GPU_HOST_PFIFO_SCHED_ERROR, INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pfifo_chsw_error",
					GPU_HOST_PFIFO_CHSW_ERROR, INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pfifo_memop_error",
					GPU_HOST_PFIFO_MEMOP_TIMEOUT_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pfifo_lb_error",
					GPU_HOST_PFIFO_LB_ERROR, INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbus_squash_error",
					GPU_HOST_PBUS_SQUASH_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbus_fecs_error",
					GPU_HOST_PBUS_FECS_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbus_timeout_error",
					GPU_HOST_PBUS_TIMEOUT_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbdma_timeout_error",
					GPU_HOST_PBDMA_TIMEOUT_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbdma_extra_error",
					GPU_HOST_PBDMA_EXTRA_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbdma_gpfifo_pb_error",
					GPU_HOST_PBDMA_GPFIFO_PB_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbdma_method_error",
					GPU_HOST_PBDMA_METHOD_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbdma_signature_error",
					GPU_HOST_PBDMA_SIGNATURE_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbdma_hce_error",
					GPU_HOST_PBDMA_HCE_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pbdma_preempt_error",
					GPU_HOST_PBDMA_PREEMPT_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("pfifo_ctxsw_timeout",
					GPU_HOST_PFIFO_CTXSW_TIMEOUT_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pfifo_fb_flush_timeout",
					GPU_HOST_PFIFO_FB_FLUSH_TIMEOUT_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_host_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "sm",
		.hw_unit = (u32)NVGPU_ERR_MODULE_SM,
		.num_instances = 8U,
		.num_errs = 21U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_NONCRITERR("l1_tag_ecc_corrected",
					GPU_SM_L1_TAG_ECC_CORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("l1_tag_ecc_uncorrected",
					GPU_SM_L1_TAG_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("cbu_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("cbu_ecc_uncorrected",
					GPU_SM_CBU_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("lrf_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("lrf_ecc_uncorrected",
					GPU_SM_LRF_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("l1_data_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("l1_data_ecc_uncorrected",
					GPU_SM_L1_DATA_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("icache_l0_data_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("icache_l0_data_ecc_uncorrected",
					GPU_SM_ICACHE_L0_DATA_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("icache_l1_data_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("icache_l1_data_ecc_uncorrected",
					GPU_SM_ICACHE_L1_DATA_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("icache_l0_predecode_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("icache_l0_predecode_ecc_uncorrected",
					GPU_SM_ICACHE_L0_PREDECODE_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("l1_tag_miss_fifo_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("l1_tag_miss_fifo_ecc_uncorrected",
					GPU_SM_L1_TAG_MISS_FIFO_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("l1_tag_s2r_pixprf_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("l1_tag_s2r_pixprf_ecc_uncorrected",
					GPU_SM_L1_TAG_S2R_PIXPRF_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("machine_check_error",
					GPU_SM_MACHINE_CHECK_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("icache_l1_predecode_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("icache_l1_predecode_ecc_uncorrected",
					GPU_SM_ICACHE_L1_PREDECODE_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "fecs",
		.hw_unit = (u32)NVGPU_ERR_MODULE_FECS,
		.num_instances = 1U,
		.num_errs = 8U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_NONCRITERR("falcon_imem_ecc_corrected",
					GPU_FECS_FALCON_IMEM_ECC_CORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("falcon_imem_ecc_uncorrected",
					GPU_FECS_FALCON_IMEM_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("falcon_dmem_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("falcon_dmem_ecc_uncorrected",
					GPU_FECS_FALCON_DMEM_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("ctxsw_watchdog_timeout",
					GPU_FECS_CTXSW_WATCHDOG_TIMEOUT,
					INJECT_SW,
					NULL, nvgpu_inject_ctxsw_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("ctxsw_crc_mismatch",
					GPU_FECS_CTXSW_CRC_MISMATCH,
					INJECT_SW,
					NULL, nvgpu_inject_ctxsw_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("fault_during_ctxsw",
					GPU_FECS_FAULT_DURING_CTXSW,
					INJECT_SW,
					NULL, nvgpu_inject_ctxsw_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("ctxsw_init_error",
					GPU_FECS_CTXSW_INIT_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_ctxsw_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "gpccs",
		.hw_unit = (u32)NVGPU_ERR_MODULE_GPCCS,
		.num_instances = 1U,
		.num_errs = 4U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_NONCRITERR("falcon_imem_ecc_corrected",
					GPU_GPCCS_FALCON_IMEM_ECC_CORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("falcon_imem_ecc_uncorrected",
					GPU_GPCCS_FALCON_IMEM_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("falcon_dmem_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("falcon_dmem_ecc_uncorrected",
					GPU_GPCCS_FALCON_DMEM_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "mmu",
		.hw_unit = (u32)NVGPU_ERR_MODULE_MMU,
		.num_instances = 1U,
		.num_errs = 4U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_NONCRITERR("l1tlb_sa_data_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("l1tlb_sa_data_ecc_uncorrected",
					GPU_MMU_L1TLB_SA_DATA_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("l1tlb_fa_data_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("l1tlb_fa_data_ecc_uncorrected",
					GPU_MMU_L1TLB_FA_DATA_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "gcc",
		.hw_unit = (u32)NVGPU_ERR_MODULE_GCC,
		.num_instances = 1U,
		.num_errs = 2U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_NONCRITERR("l15_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("l15_ecc_uncorrected",
					GPU_GCC_L15_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "pmu",
		.hw_unit = (u32)NVGPU_ERR_MODULE_PMU,
		.num_instances = 1U,
		.num_errs = 5U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_NONCRITERR("falcon_imem_ecc_corrected",
					GPU_PMU_FALCON_IMEM_ECC_CORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("falcon_imem_ecc_uncorrected",
					GPU_PMU_FALCON_IMEM_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("falcon_dmem_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("falcon_dmem_ecc_uncorrected",
					GPU_PMU_FALCON_DMEM_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("bar0_error_timeout",
					GPU_PMU_BAR0_ERROR_TIMEOUT, INJECT_SW,
					NULL, nvgpu_inject_pmu_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "pgraph",
		.hw_unit = (u32)NVGPU_ERR_MODULE_PGRAPH,
		.num_instances = 1U,
		.num_errs = 12U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_CRITERR("fe_exception",
					GPU_PGRAPH_FE_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("memfmt_exception",
					GPU_PGRAPH_MEMFMT_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pd_exception",
					GPU_PGRAPH_PD_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("scc_exception",
					GPU_PGRAPH_SCC_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("ds_exception",
					GPU_PGRAPH_DS_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("ssync_exception",
					GPU_PGRAPH_SSYNC_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("mme_exception",
					GPU_PGRAPH_MME_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("sked_exception",
					GPU_PGRAPH_SKED_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("be_exception",
					GPU_PGRAPH_BE_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("mpc_exception",
					GPU_PGRAPH_MPC_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("illegal_error",
					GPU_PGRAPH_ILLEGAL_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("gpc_gfx_exception",
					GPU_PGRAPH_GPC_GFX_EXCEPTION,
					INJECT_SW,
					NULL, nvgpu_inject_gr_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "ltc",
		.hw_unit = (u32)NVGPU_ERR_MODULE_LTC,
		.num_instances = 1U,
		.num_errs = 8U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_NONCRITERR("cache_dstg_ecc_corrected",
					GPU_LTC_CACHE_DSTG_ECC_CORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("cache_dstg_ecc_uncorrected",
					GPU_LTC_CACHE_DSTG_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("cache_tstg_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("cache_tstg_ecc_uncorrected",
					GPU_LTC_CACHE_TSTG_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("cache_rstg_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("cache_rstg_ecc_uncorrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("cache_dstg_be_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("cache_dstg_be_ecc_uncorrected",
					GPU_LTC_CACHE_DSTG_BE_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "hubmmu",
		.hw_unit = (u32)NVGPU_ERR_MODULE_HUBMMU,
		.num_instances = 1U,
		.num_errs = 9U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_NONCRITERR("hubmmu_l2tlb_sa_data_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("hubmmu_l2tlb_sa_data_ecc_uncorrected",
					GPU_HUBMMU_L2TLB_SA_DATA_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("hubmmu_tlb_sa_data_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("hubmmu_tlb_sa_data_ecc_uncorrected",
					GPU_HUBMMU_TLB_SA_DATA_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("hubmmu_pte_data_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("hubmmu_pte_data_ecc_uncorrected",
					GPU_HUBMMU_PTE_DATA_ECC_UNCORRECTED,
					INJECT_TYPE,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("hubmmu_pde0_data_ecc_corrected",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("hubmmu_pde0_data_ecc_uncorrected",
					GPU_HUBMMU_PDE0_DATA_ECC_UNCORRECTED,
					INJECT_SW,
					NULL, nvgpu_inject_ecc_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("hubmmu_page_fault_error",
					GPU_HUBMMU_PAGE_FAULT_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_mmu_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "pri",
		.hw_unit = (u32)NVGPU_ERR_MODULE_PRI,
		.num_instances = 1U,
		.num_errs = 2U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_CRITERR("pri_timeout_error",
					GPU_PRI_TIMEOUT_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_pri_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("pri_access_violation",
					GPU_PRI_ACCESS_VIOLATION,
					INJECT_SW,
					NULL, nvgpu_inject_pri_swerror,
					NULL, NULL, 0, 0),
		},
	},
	{
		.name = "ce",
		.hw_unit = (u32)NVGPU_ERR_MODULE_CE,
		.num_instances = 1U,
		.num_errs = 5U,
		.errs = (struct nvgpu_err_desc[]) {
			GPU_CRITERR("ce_launch_error",
					GPU_CE_LAUNCH_ERROR,
					INJECT_SW,
					NULL, nvgpu_inject_ce_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("ce_block_pipe",
					GPU_CE_BLOCK_PIPE,
					INJECT_SW,
					NULL, nvgpu_inject_ce_swerror,
					NULL, NULL, 0, 0),
			GPU_NONCRITERR("ce_nonblock_pipe",
					0, INJECT_NONE,
					NULL, NULL,
					NULL, NULL, 0, 0),
			GPU_CRITERR("ce_invalid_config",
					GPU_CE_INVALID_CONFIG,
					INJECT_SW,
					NULL, nvgpu_inject_ce_swerror,
					NULL, NULL, 0, 0),
			GPU_CRITERR("ce_method_buffer_fault",
					GPU_CE_METHOD_BUFFER_FAULT,
					INJECT_SW,
					NULL, nvgpu_inject_ce_swerror,
					NULL, NULL, 0, 0),
		},
	},
};

u32 size_of_gv11b_lut = sizeof(gv11b_err_lut) /
					sizeof(struct nvgpu_err_hw_module);

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

#include <stdio.h>
#include <stdlib.h>

#include <nvgpu/hw/ga10b/hw_gr_ga10b.h>
#include <nvgpu/hw/ga10b/hw_ce_ga10b.h>
#include <nvgpu/hw/gk20a/hw_bus_gk20a.h>
#include <nvgpu/hw/ga10b/hw_ltc_ga10b.h>
#include <nvgpu/hw/ga10b/hw_top_ga10b.h>
#include <nvgpu/hw/ga10b/hw_proj_ga10b.h>
#include <nvgpu/hw/ga10b/hw_pbdma_ga10b.h>
#include <nvgpu/hw/ga10b/hw_runlist_ga10b.h>

#define NVGPU_CHANNEL_SIZE 512
#define NVGPU_RUNLIST_SIZE 4

void nvgpu_posix_bug(const char *msg, int line_no)
{
	printf("%s:%d BUG detected!", msg, line_no);
        exit(1);
}

void nvgpu_checker_insert_reg_data_with_mask(FILE *header,
        const char *reg_name, int reg_arg, u32 reg_offset, u32 reg_value,
	u32 reg_mask)
{
        fprintf(header, "       { 0x%08x, 0x%08x, 0x%08x }, /* %s(",
                reg_offset, reg_value, reg_mask, reg_name);

	/* NVGPU defines registers as function macros in hw headers.
	 * It may or maynot have arguments
	 */
	if (reg_arg >= 0) {
		fprintf(header, "%d", reg_arg);
	}
	fprintf(header, ") */\n");
}

void nvgpu_checker_insert_reg_data(FILE *header,
	const char *reg_name, int reg_arg, u32 reg_offset, u32 reg_value)
{
	return nvgpu_checker_insert_reg_data_with_mask(header, reg_name,
			reg_arg, reg_offset, reg_value, 0xFFFFFFFF);
}

void nvgpu_checker_print_intr_bus_reg_list(FILE *header)
{
	nvgpu_checker_insert_reg_data(header,
		"bus_intr_en_1_r", -1, bus_intr_en_1_r(), 0x0);
}

void nvgpu_checker_print_pbdma_reg_list(FILE *header)
{
	u32 pbdma_id, tree = 0U;
	u32 pbdma_counter = proj_host_num_pbdma_v();

	for (pbdma_id = 0; pbdma_id < pbdma_counter; pbdma_id++) {
		nvgpu_checker_insert_reg_data(header,
			"pbdma_intr_0_r", pbdma_id,
			pbdma_intr_0_r(pbdma_id), 0x0);
		nvgpu_checker_insert_reg_data(header,
			"pbdma_intr_1_r", pbdma_id,
			pbdma_intr_1_r(pbdma_id), 0x0);
		nvgpu_checker_insert_reg_data(header,
			"pbdma_intr_0_en_set_tree_r", pbdma_id,
                        pbdma_intr_0_en_set_tree_r(pbdma_id, tree),
			0xcfafe000);
		nvgpu_checker_insert_reg_data(header,
			"pbdma_intr_1_en_set_tree_r", pbdma_id,
                        pbdma_intr_1_en_set_tree_r(pbdma_id, tree),
			0x8000001f);
	}
}

void nvgpu_checker_print_runlist_reg_list(FILE *header)
{
	u32 intr_tree_0 = 0U, intr_tree_1 = 1U;
	u32 runlist_id, runlist_pri_base = 0U;

	u32 runlist_pri_base_on_runlist[NVGPU_RUNLIST_SIZE] = {
		0xc00000, 0xc00400, 0xc00c00, 0xc00800
	};

	u32 runlist_intr_values[NVGPU_RUNLIST_SIZE] = {
		0xf0, 0x30, 0x0, 0x30
	};

	u32 runlist_intr_vector_0_values[NVGPU_RUNLIST_SIZE] = {
		0x800000a0, 0x800000a1, 0x800000a2, 0x800000a3
	};

	u32 runlist_intr_vector_1_values[NVGPU_RUNLIST_SIZE] = {
		0xe0, 0xe1, 0xe2, 0xe3
	};

	for (runlist_id = 0U; runlist_id < NVGPU_RUNLIST_SIZE; runlist_id++) {
		runlist_pri_base = runlist_pri_base_on_runlist[runlist_id];

		nvgpu_checker_insert_reg_data(header,
                        "runlist_intr_0_en_set_tree_r", intr_tree_0,
			nvgpu_safe_add_u32(runlist_pri_base,
			runlist_intr_0_en_set_tree_r(intr_tree_0)), 0x31007);

		nvgpu_checker_insert_reg_data(header,
                        "runlist_intr_0_en_set_tree_r", intr_tree_1,
			nvgpu_safe_add_u32(runlist_pri_base,
			runlist_intr_0_en_set_tree_r(intr_tree_1)), 0x0);

		nvgpu_checker_insert_reg_data(header,
                        "runlist_intr_0_en_clear_tree_r", intr_tree_0,
			nvgpu_safe_add_u32(runlist_pri_base,
			runlist_intr_0_en_clear_tree_r(intr_tree_0)), 0x31007);

		nvgpu_checker_insert_reg_data(header,
                        "runlist_intr_0_en_clear_tree_r", intr_tree_1,
			nvgpu_safe_add_u32(runlist_pri_base,
			runlist_intr_0_en_clear_tree_r(intr_tree_1)), 0x0);

		nvgpu_checker_insert_reg_data(header, "runlist_intr_0_r", -1,
			nvgpu_safe_add_u32(runlist_pri_base,
			runlist_intr_0_r()),
			runlist_intr_values[runlist_id]);

		nvgpu_checker_insert_reg_data(header,
                        "runlist_intr_vectorid_r", intr_tree_0,
			nvgpu_safe_add_u32(runlist_pri_base,
			runlist_intr_vectorid_r(intr_tree_0)),
			runlist_intr_vector_0_values[runlist_id]);

		nvgpu_checker_insert_reg_data(header,
                        "runlist_intr_vectorid_r", intr_tree_1,
			nvgpu_safe_add_u32(runlist_pri_base,
			runlist_intr_vectorid_r(intr_tree_1)),
			runlist_intr_vector_1_values[runlist_id]);
	}
}

void nvgpu_checker_print_ce_lce_reg_list(FILE *header)
{
	u32 inst_id = 0; /* no MIG support */

	nvgpu_checker_insert_reg_data(header,
		"ce_lce_intr_en_r", inst_id,
                ce_lce_intr_en_r(inst_id), 0x4);
	nvgpu_checker_insert_reg_data(header,
                "ce_lce_intr_ctrl_r", inst_id,
                ce_lce_intr_ctrl_r(inst_id), 0x800000c2);
	nvgpu_checker_insert_reg_data(header,
                "ce_lce_intr_notify_ctrl_r", inst_id,
                ce_lce_intr_notify_ctrl_r(inst_id), 0x80000000);
}

void nvgpu_checker_print_gr_reg_list(FILE *header)
{
	nvgpu_checker_insert_reg_data(header,
		"gr_intr_en_r", -1,
                gr_intr_en_r(), 0x780150);
	nvgpu_checker_insert_reg_data(header,
		"gr_exception_en_r", -1,
                gr_exception_en_r(), 0x10003bf);
	nvgpu_checker_insert_reg_data(header,
		"gr_exception1_en_r", -1,
                gr_exception1_en_r(), 0x3);
}

void nvgpu_checker_print_ltc_reg_list(FILE *header)
{
	nvgpu_checker_insert_reg_data(header,
		"ltc_ltcs_ltss_intr_r", -1,
                ltc_ltcs_ltss_intr_r(), 0xfb2f0000);
	nvgpu_checker_insert_reg_data(header,
		"ltc_ltcs_ltss_intr2_r", -1,
                ltc_ltcs_ltss_intr2_r(), 0xffff0000);
	nvgpu_checker_insert_reg_data(header,
		"ltc_ltcs_ltss_intr3_r", -1,
                ltc_ltcs_ltss_intr3_r(), 0x7c7f0000);
}

void nvgpu_checker_print_channel_reg_list(FILE *header)
{
	u32 runlist_id, channel_id;
	u32 chram_bar0_offset_on_runlist[NVGPU_RUNLIST_SIZE] = {
		0xc20000, 0xc22000, 0xc24000, 0xc26000
	};

	for (runlist_id = 0U; runlist_id < NVGPU_RUNLIST_SIZE; runlist_id++) {
		for (channel_id = 0U; channel_id < NVGPU_CHANNEL_SIZE; channel_id++) {
			nvgpu_checker_insert_reg_data_with_mask(header,
                                "runlist_chram_channel_r", channel_id,
                                nvgpu_safe_add_u32(chram_bar0_offset_on_runlist[runlist_id],
				runlist_chram_channel_r(channel_id)), 0x0, 0x00001F00);
		}
	}
}

void nvgpu_checker_generate_hw_registers_table_begin(FILE *header)
{
        fprintf(header, "/*\n");
        fprintf(header, " * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.\n");
        fprintf(header, " *\n");
        fprintf(header, " * NVIDIA Corporation and its licensors retain all intellectual property\n");
        fprintf(header, " * and proprietary rights in and to this software, related documentation\n");
        fprintf(header, " * and any modifications thereto.  Any use, reproduction, disclosure or\n");
        fprintf(header, " * distribution of this software and related documentation without an express\n");
        fprintf(header, " * license agreement from NVIDIA Corporation is strictly prohibited.\n");
        fprintf(header, " *\n");
        fprintf(header, " * This is a generated file. Do not edit.\n");
        fprintf(header, " *\n");
        fprintf(header, " * Steps to regenerate:\n");
        fprintf(header, " *  cd $TEGRA_TOP/kernel/nvgpu/scripts/checker/hw_register_generator\n");
        fprintf(header, " *  make\n");
        fprintf(header, " *  make generate\n");
        fprintf(header, " */\n\n");

        fprintf(header, "#include \"hw_registers_checker.h\"\n\n");
        fprintf(header, "const struct hw_register_set hw_registers[] = {\n");
}

void nvgpu_checker_generate_hw_registers_table_data(FILE *header)
{
	nvgpu_checker_print_intr_bus_reg_list(header);
	nvgpu_checker_print_pbdma_reg_list(header);
	nvgpu_checker_print_runlist_reg_list(header);
	nvgpu_checker_print_ce_lce_reg_list(header);
	nvgpu_checker_print_gr_reg_list(header);
	nvgpu_checker_print_ltc_reg_list(header);
	nvgpu_checker_print_channel_reg_list(header);
}

void nvgpu_checker_generate_hw_registers_table_end(FILE *header)
{
        fprintf(header, "};\n\n");

        fprintf(header, "u32 hw_register_set_size()\n");
        fprintf(header, "{\n");
        fprintf(header, "       return ((sizeof(hw_registers) /\n");
        fprintf(header, "			sizeof(struct hw_register_set)));\n");
        fprintf(header, "}\n");
}

int main(int argc, char *argv[])
{
        FILE *header = fopen(argv[1], "w");

	nvgpu_checker_generate_hw_registers_table_begin(header);
	nvgpu_checker_generate_hw_registers_table_data(header);
	nvgpu_checker_generate_hw_registers_table_end(header);

        fclose(header);

        return 0;
}
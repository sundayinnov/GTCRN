#ifndef GTCRN_WEIGHTS_COMPAT_H
#define GTCRN_WEIGHTS_COMPAT_H

#include "gtcrn_weights.h"

/* ---- 基础卷积映射 ---- */
#define encoder_conv0_w  encoder_en_convs_0_conv_weight
#define encoder_conv0_b  encoder_en_convs_0_conv_bias
#define encoder_conv1_w  encoder_en_convs_1_conv_weight
#define encoder_conv1_b  encoder_en_convs_1_conv_bias

/* ---- Encoder GTBlock 2 ---- */
#define eb2_pc1_w encoder_en_convs_2_point_conv1_weight
#define eb2_pc1_b encoder_en_convs_2_point_conv1_bias
#define eb2_dc_w  encoder_en_convs_2_depth_conv_weight
#define eb2_dc_b  encoder_en_convs_2_depth_conv_bias
#define eb2_pc2_w encoder_en_convs_2_point_conv2_weight
#define eb2_pc2_b encoder_en_convs_2_point_conv2_bias
#define eb2_attn_wih val_188
#define eb2_attn_whh val_189
#define eb2_attn_b   val_205
#define eb2_attn_fc_w val_221
#define eb2_attn_fc_b encoder_en_convs_2_tra_att_fc_bias

/* ---- Encoder GTBlock 3 ---- */
#define eb3_pc1_w encoder_en_convs_3_point_conv1_weight
#define eb3_pc1_b encoder_en_convs_3_point_conv1_bias
#define eb3_dc_w  encoder_en_convs_3_depth_conv_weight
#define eb3_dc_b  encoder_en_convs_3_depth_conv_bias
#define eb3_pc2_w encoder_en_convs_3_point_conv2_weight
#define eb3_pc2_b encoder_en_convs_3_point_conv2_bias
#define eb3_attn_wih val_320
#define eb3_attn_whh val_321
#define eb3_attn_b   val_337
#define eb3_attn_fc_w val_350
#define eb3_attn_fc_b encoder_en_convs_3_tra_att_fc_bias

/* ---- Encoder GTBlock 4 ---- */
#define eb4_pc1_w encoder_en_convs_4_point_conv1_weight
#define eb4_pc1_b encoder_en_convs_4_point_conv1_bias
#define eb4_dc_w  encoder_en_convs_4_depth_conv_weight
#define eb4_dc_b  encoder_en_convs_4_depth_conv_bias
#define eb4_pc2_w encoder_en_convs_4_point_conv2_weight
#define eb4_pc2_b encoder_en_convs_4_point_conv2_bias
#define eb4_attn_wih val_449
#define eb4_attn_whh val_450
#define eb4_attn_b   val_466
#define eb4_attn_fc_w val_479
#define eb4_attn_fc_b encoder_en_convs_4_tra_att_fc_bias

/* ---- Decoder GTBlock 0 ---- */
#define db0_pc1_w decoder_de_convs_0_point_conv1_weight
#define db0_pc1_b decoder_de_convs_0_point_conv1_bias
#define db0_dc_w  decoder_de_convs_0_depth_conv_weight
#define db0_dc_b  decoder_de_convs_0_depth_conv_bias
#define db0_pc2_w decoder_de_convs_0_point_conv2_weight
#define db0_pc2_b decoder_de_convs_0_point_conv2_bias
#define db0_attn_wih val_1178
#define db0_attn_whh val_1179
#define db0_attn_b   val_1195
#define db0_attn_fc_w val_1208
#define db0_attn_fc_b decoder_de_convs_0_tra_att_fc_bias

/* ---- Decoder GTBlock 1 ---- */
#define db1_pc1_w decoder_de_convs_1_point_conv1_weight
#define db1_pc1_b decoder_de_convs_1_point_conv1_bias
#define db1_dc_w  decoder_de_convs_1_depth_conv_weight
#define db1_dc_b  decoder_de_convs_1_depth_conv_bias
#define db1_pc2_w decoder_de_convs_1_point_conv2_weight
#define db1_pc2_b decoder_de_convs_1_point_conv2_bias
#define db1_attn_wih val_1307
#define db1_attn_whh val_1308
#define db1_attn_b   val_1324
#define db1_attn_fc_w val_1337
#define db1_attn_fc_b decoder_de_convs_1_tra_att_fc_bias

/* ---- Decoder GTBlock 2 ---- */
#define db2_pc1_w decoder_de_convs_2_point_conv1_weight
#define db2_pc1_b decoder_de_convs_2_point_conv1_bias
#define db2_dc_w  decoder_de_convs_2_depth_conv_weight
#define db2_dc_b  decoder_de_convs_2_depth_conv_bias
#define db2_pc2_w decoder_de_convs_2_point_conv2_weight
#define db2_pc2_b decoder_de_convs_2_point_conv2_bias
#define db2_attn_wih val_1436
#define db2_attn_whh val_1437
#define db2_attn_b   val_1453
#define db2_attn_fc_w val_1466
#define db2_attn_fc_b decoder_de_convs_2_tra_att_fc_bias

/* ---- Decoder Conv ---- */
#define decoder_conv3_w decoder_de_convs_3_conv_weight
#define decoder_conv3_b decoder_de_convs_3_conv_bias
#define decoder_conv4_w decoder_de_convs_4_conv_weight
#define decoder_conv4_b decoder_de_convs_4_conv_bias

/* ---- DPGRNN 补充符号映射 ---- */
#define val_679 dpgrnn1_intra_fc_bias
#define val_685 dpgrnn1_intra_ln_weight
#define val_686 dpgrnn1_intra_ln_bias
#define val_794 dpgrnn1_inter_fc_bias
#define val_800 dpgrnn1_inter_ln_weight
#define val_801 dpgrnn1_inter_ln_bias

#define val_979 dpgrnn2_intra_fc_bias
#define val_985 dpgrnn2_intra_ln_weight
#define val_986 dpgrnn2_intra_ln_bias
#define val_1094 dpgrnn2_inter_fc_bias
#define val_1100 dpgrnn2_inter_ln_weight
#define val_1101 dpgrnn2_inter_ln_bias

#define encoder_conv2_w  encoder_en_convs_1_conv_weight
#define encoder_conv2_b  encoder_en_convs_1_conv_bias

#endif
/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t ATTENTION_OUT_INDEX = 0;
constexpr size_t SOFTMAX_LSE_INDEX = 1;
constexpr size_t DIM_THREE = 3;
constexpr size_t DIM_FOUR = 4;

ge::graphStatus InferShapeSparseFlashAttention(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashAttention", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;
    gert::Shape *softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    OPS_LOG_E_IF_NULL(context, softmaxLseShape, return ge::GRAPH_FAILED)
    if (queryShape->GetDimNum() == DIM_THREE) {
        // TND: query is [T, N, D], LSE is [T, N].
        softmaxLseShape->SetDimNum(2);
        softmaxLseShape->SetDim(0, queryShape->GetDim(0));
        softmaxLseShape->SetDim(1, queryShape->GetDim(1));
    } else if (queryShape->GetDimNum() == DIM_FOUR) {
        // BSND: query is [B, S, N, D], LSE is [B, N, S].
        softmaxLseShape->SetDimNum(3);
        softmaxLseShape->SetDim(0, queryShape->GetDim(0));
        softmaxLseShape->SetDim(1, queryShape->GetDim(2));
        softmaxLseShape->SetDim(2, queryShape->GetDim(1));
    } else {
        OPS_LOG_E("SparseFlashAttention", "query dim num should be 3 or 4.");
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseFlashAttention(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashAttention", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(ATTENTION_OUT_INDEX, inputDataType);
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseFlashAttention).InferShape(InferShapeSparseFlashAttention).InferDataType(InferDataTypeSparseFlashAttention);
} // namespace ops
  

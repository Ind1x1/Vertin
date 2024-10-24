// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// Sonnet Vertin

#include <torch/extension.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>

#include "cpu_vertin.h"
#include "workspace.hpp"

using namespace std::string_literals;
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

void Vertin_Optimizer::Step_1(float* grads,
                              float* _exp_avg,
                              float* _exp_avg_sq,
                              size_t _param_size,
                              bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size, grads, _exp_avg, _exp_avg_sq, _param_size, half_precision);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        // float step_size = -1 * _alpha / _bias_correction1;
        // float w_decay = -1 * _alpha * _weight_decay;

        for(size_t t = rounded_size; t < _param_size; t += TILE){
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = (float)grads[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];

                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
        }
    }
}

void Vertin_Optimizer::Step_4(float* grads,
                              float* _exp_avg,
                              float* _exp_avg_sq,
                              size_t _param_size,
                              bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size, grads, _exp_avg, _exp_avg_sq, _param_size, half_precision);
#endif
    if (_param_size > rounded_size)
        Step_1((grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               half_precision);
}

int create_vertin_optimizer(int optimizer_id,
                            float alpha,
                            float betta1,
                            float betta2,
                            float eps,
                            float weight_decay,
                            bool adamw_mode,
                            bool should_log)
{
    auto opt = 
        std::make_shared<Vertin_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }
    return 0;
}

void Vertin_Optimizer::Step_8(float* grads,
                              float* _exp_avg,
                              float* _exp_avg_sq,
                              size_t _param_size,
                              bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size, grads, _exp_avg, _exp_avg_sq, _param_size, half_precision);
#endif
    if (_param_size > rounded_size)
        Step_4((grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               half_precision);
}

int sn_vertin_step(int optimizer_id,
                   size_t step,
                   float lr,
                   float beta1,
                   float beta2,
                   float epsilon,
                   float weight_decay,
                   bool bias_correction,
                   torch::Tensor& grads,
                   torch::Tensor& exp_avg,
                   torch::Tensor& exp_avg_sq)
{
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Vertin_Optimizer> opt =
        std::static_pointer_cast<Vertin_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    opt->Step_8(grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                grads_c.numel(),
                false);
    return 0;
}

int destroy_vertin_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}
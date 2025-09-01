#pragma once
#define INPUT_SIZE 14
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 7
#define TIMESTEPS 8
#define BETA 0.9f
#define THRESHOLD 1.0f

void snn_inference(const float in[INPUT_SIZE], float out_norm[OUTPUT_SIZE]);

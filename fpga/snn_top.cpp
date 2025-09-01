#include "snn_top.h"
#include "fc1_weights.h"
#include "fc1_bias.h"
#include "fc2_weights.h"
#include "fc2_bias.h"

void snn_inference(const float in[INPUT_SIZE], float out_norm[OUTPUT_SIZE]) {
    float mem1[HIDDEN_SIZE] = {0.0f};
    float out_accum[OUTPUT_SIZE] = {0.0f};

    for (int t = 0; t < TIMESTEPS; t++) {
        float hidden_spike[HIDDEN_SIZE];

        // FC1
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float sum = fc1_bias[i];
            for (int j = 0; j < INPUT_SIZE; j++) {
                sum += fc1_weights[i][j] * in[j];
            }
            mem1[i] = BETA * mem1[i] + sum;
            if (mem1[i] >= THRESHOLD) {
                hidden_spike[i] = 1.0f;
                mem1[i] = 0.0f;
            } else {
                hidden_spike[i] = 0.0f;
            }
        }

        // FC2
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            float sum = fc2_bias[k];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                sum += fc2_weights[k][i] * hidden_spike[i];
            }
            out_accum[k] += sum;
        }
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        out_norm[k] = out_accum[k] / (float)TIMESTEPS;
    }
}


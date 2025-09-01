#include <stdio.h>
#include "snn_top.h"     // ONLY the header, not the .cpp
#include "sample_input.h"
#include "norm.h"

int main() {
    float out_norm[OUTPUT_SIZE];
    snn_inference(sample_input, out_norm);

    printf("Normalized output:\n");
    for (int k = 0; k < OUTPUT_SIZE; k++) printf("%f ", out_norm[k]);
    printf("\n");

    // Denormalize and print
    printf("Denormalized output:\n");
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        float out_denorm = out_norm[k] * y_std[k] + y_mean[k];
        printf("%f ", out_denorm);
    }
    printf("\n");
    return 0;
}

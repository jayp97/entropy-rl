#include <time.h>
#include "entropy.h"

void performance_test() {
    long test_time = 10;
    Entropy env = {
        .width = 800,
        .height = 600,
        .difficulty = DIFF_EASY,
        .agent_role_config = -1,
        .reward_invalid = -0.1f,
        .reward_palindrome_delta = 0.0f,
        .rng_state = 42,
    };

    env.observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % NUM_ACTIONS;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
}

int main() {
    performance_test();
    return 0;
}

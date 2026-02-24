#include "entropy.h"
#define Env Entropy
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->difficulty = (int)unpack(kwargs, "difficulty");
    env->agent_role_config = (int)unpack(kwargs, "agent_role");
    env->reward_invalid = unpack(kwargs, "reward_invalid");
    env->reward_palindrome_delta = unpack(kwargs, "reward_palindrome_delta");
    env->width = (int)unpack(kwargs, "width");
    env->height = (int)unpack(kwargs, "height");
    env->rng_state = (unsigned int)unpack(kwargs, "seed");
    if (env->rng_state == 0) env->rng_state = 42;
    env->client = NULL;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "order_score", log->order_score);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

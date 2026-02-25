#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

// Board constants
#define BOARD_SIZE 7
#define NUM_CELLS 49
#define NUM_COLORS 7
#define EMPTY_CELL 0

// Action space: 49 chaos placements + 196 order slides + 1 order pass = 246
#define NUM_ACTIONS 246
#define CHAOS_ACTION_BASE 0
#define ORDER_ACTION_BASE 49
#define ORDER_PASS_ACTION 245

// Observation size: 7*49 color planes + 49 empty mask + 7 bag + 3 metadata + 246 action masks = 648
#define OBS_SIZE 648

// Directions for Order slide: N, S, E, W
#define DIR_N 0
#define DIR_S 1
#define DIR_E 2
#define DIR_W 3

// Roles
#define ROLE_ORDER 0
#define ROLE_CHAOS 1

// Difficulty levels
#define DIFF_EASY 0
#define DIFF_MEDIUM 1
#define DIFF_HARD 2

// Required struct. Only use floats!
typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float order_score;
    float n;
} Log;

typedef struct Client Client;

typedef struct {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    // Board state
    int board[NUM_CELLS];           // 0=empty, 1-7=colors
    int bag[NUM_COLORS];            // remaining count per color
    int total_remaining;            // total pieces left in bag

    // Turn state
    int current_draw;               // color Chaos just drew (1-7), 0 if none
    int turn_number;                // 0-48, which placement we're on

    // Role tracking
    int agent_role;                 // 0=Order, 1=Chaos (current episode)
    int agent_role_config;          // -1=random, 0=Order, 1=Chaos

    // Scoring & rewards
    float reward_invalid;
    float reward_palindrome_delta;

    // Opponent difficulty
    int difficulty;

    // Rendering
    Client* client;
    int width;
    int height;

    int tick;
    unsigned int rng_state;         // Per-environment RNG
} Entropy;

// ==================== RNG ====================

static unsigned int entropy_rand(Entropy* env) {
    env->rng_state = env->rng_state * 1103515245 + 12345;
    return (env->rng_state >> 16) & 0x7FFF;
}

// ==================== Palindrome Scoring ====================

int score_line(int line[BOARD_SIZE]) {
    int total = 0;
    for (int start = 0; start < BOARD_SIZE; start++) {
        for (int end = start + 1; end < BOARD_SIZE; end++) {
            int len = end - start + 1;
            int is_palindrome = 1;
            for (int k = 0; k < len / 2; k++) {
                if (line[start + k] != line[end - k]) {
                    is_palindrome = 0;
                    break;
                }
            }
            if (is_palindrome) {
                total += len;
            }
        }
    }
    return total;
}

int compute_total_score(Entropy* env) {
    int total = 0;
    int line[BOARD_SIZE];

    // Score all 7 rows
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            line[c] = env->board[r * BOARD_SIZE + c];
        }
        total += score_line(line);
    }

    // Score all 7 columns
    for (int c = 0; c < BOARD_SIZE; c++) {
        for (int r = 0; r < BOARD_SIZE; r++) {
            line[r] = env->board[r * BOARD_SIZE + c];
        }
        total += score_line(line);
    }

    return total;
}

// Score just the row and column containing a specific cell
int score_affected_lines(Entropy* env, int cell) {
    int row = cell / BOARD_SIZE;
    int col = cell % BOARD_SIZE;
    int line[BOARD_SIZE];
    int total = 0;

    // Score the row
    for (int c = 0; c < BOARD_SIZE; c++) {
        line[c] = env->board[row * BOARD_SIZE + c];
    }
    total += score_line(line);

    // Score the column
    for (int r = 0; r < BOARD_SIZE; r++) {
        line[r] = env->board[r * BOARD_SIZE + col];
    }
    total += score_line(line);

    return total;
}

// ==================== Piece Drawing ====================

void draw_piece(Entropy* env) {
    if (env->total_remaining <= 0) {
        env->current_draw = 0;
        return;
    }
    int r = entropy_rand(env) % env->total_remaining;
    int cumulative = 0;
    for (int c = 0; c < NUM_COLORS; c++) {
        cumulative += env->bag[c];
        if (r < cumulative) {
            env->current_draw = c + 1;  // colors are 1-indexed
            env->bag[c]--;
            env->total_remaining--;
            return;
        }
    }
}

// ==================== Order Slide Mechanics ====================

int compute_slide_destination(Entropy* env, int cell, int direction) {
    int row = cell / BOARD_SIZE;
    int col = cell % BOARD_SIZE;
    int dr = 0, dc = 0;
    switch (direction) {
        case DIR_N: dr = -1; break;
        case DIR_S: dr = 1;  break;
        case DIR_E: dc = 1;  break;
        case DIR_W: dc = -1; break;
    }
    int r = row + dr;
    int c = col + dc;
    int last_valid = cell;
    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
        if (env->board[r * BOARD_SIZE + c] != EMPTY_CELL) break;
        last_valid = r * BOARD_SIZE + c;
        r += dr;
        c += dc;
    }
    return last_valid;
}

// Execute an Order slide. Returns 1 on success, 0 on invalid.
int execute_order_slide(Entropy* env, int cell, int direction) {
    if (cell < 0 || cell >= NUM_CELLS) return 0;
    if (env->board[cell] == EMPTY_CELL) return 0;

    int dest = compute_slide_destination(env, cell, direction);
    if (dest == cell) return 0;  // no movement possible

    int color = env->board[cell];
    env->board[cell] = EMPTY_CELL;
    env->board[dest] = color;
    return 1;
}

// ==================== Action Masks ====================

// Compute which actions are valid (0.0) or invalid (1.0) for current state.
// Mask layout: [0-48] chaos placements, [49-244] order slides, [245] order pass
void compute_action_masks(Entropy* env, float* masks) {
    // Start with all masked (invalid)
    for (int i = 0; i < NUM_ACTIONS; i++) {
        masks[i] = 1.0f;
    }

    if (env->agent_role == ROLE_ORDER) {
        // Order phase: slides and pass are potentially valid
        for (int cell = 0; cell < NUM_CELLS; cell++) {
            if (env->board[cell] == EMPTY_CELL) continue;
            for (int dir = 0; dir < 4; dir++) {
                int dest = compute_slide_destination(env, cell, dir);
                if (dest != cell) {
                    // This slide would actually move the piece
                    masks[ORDER_ACTION_BASE + cell * 4 + dir] = 0.0f;
                }
            }
        }
        // Pass is always valid for Order
        masks[ORDER_PASS_ACTION] = 0.0f;
    } else {
        // Chaos phase: placements on empty cells are valid
        for (int cell = 0; cell < NUM_CELLS; cell++) {
            if (env->board[cell] == EMPTY_CELL && env->current_draw > 0) {
                masks[CHAOS_ACTION_BASE + cell] = 0.0f;
            }
        }
    }
}

// ==================== Observations ====================

void compute_observations(Entropy* env) {
    int idx = 0;

    // 7 color planes (one-hot), each 49 values
    for (int color = 1; color <= NUM_COLORS; color++) {
        for (int i = 0; i < NUM_CELLS; i++) {
            env->observations[idx++] = (env->board[i] == color) ? 1.0f : 0.0f;
        }
    }

    // Empty cell plane (49 values)
    for (int i = 0; i < NUM_CELLS; i++) {
        env->observations[idx++] = (env->board[i] == EMPTY_CELL) ? 1.0f : 0.0f;
    }

    // Bag state (7 values, normalized)
    for (int c = 0; c < NUM_COLORS; c++) {
        env->observations[idx++] = env->bag[c] / 7.0f;
    }

    // Current draw (normalized, 0 if Order phase / no draw)
    env->observations[idx++] = (env->agent_role == ROLE_CHAOS && env->current_draw > 0)
        ? env->current_draw / 7.0f : 0.0f;

    // Phase indicator: what the AGENT is doing this step
    // 0.0 = agent is Chaos (placing), 1.0 = agent is Order (sliding)
    env->observations[idx++] = (env->agent_role == ROLE_ORDER) ? 1.0f : 0.0f;

    // Turn progress
    env->observations[idx++] = env->turn_number / 49.0f;

    // Action masks (246 values): 0.0 = valid, 1.0 = invalid
    compute_action_masks(env, &env->observations[idx]);
    idx += NUM_ACTIONS;
}

// ==================== Scripted Opponents ====================

// --- Chaos Opponents (used when agent plays Order) ---

void scripted_chaos_easy(Entropy* env) {
    draw_piece(env);
    if (env->current_draw == 0) return;

    // Collect empty cells
    int empty[NUM_CELLS];
    int n_empty = 0;
    for (int i = 0; i < NUM_CELLS; i++) {
        if (env->board[i] == EMPTY_CELL) {
            empty[n_empty++] = i;
        }
    }
    if (n_empty == 0) return;

    int cell = empty[entropy_rand(env) % n_empty];
    env->board[cell] = env->current_draw;
    env->turn_number++;
}

void scripted_chaos_hard(Entropy* env) {
    draw_piece(env);
    if (env->current_draw == 0) return;

    int empty[NUM_CELLS];
    int n_empty = 0;
    for (int i = 0; i < NUM_CELLS; i++) {
        if (env->board[i] == EMPTY_CELL) {
            empty[n_empty++] = i;
        }
    }
    if (n_empty == 0) return;

    // Greedy: pick the placement that minimizes score on affected lines
    int best_cell = empty[0];
    int min_score = 999999;

    for (int e = 0; e < n_empty; e++) {
        int cell = empty[e];
        env->board[cell] = env->current_draw;
        int s = score_affected_lines(env, cell);
        env->board[cell] = EMPTY_CELL;

        if (s < min_score) {
            min_score = s;
            best_cell = cell;
        }
    }

    env->board[best_cell] = env->current_draw;
    env->turn_number++;
}

// --- Order Opponents (used when agent plays Chaos) ---

void scripted_order_easy(Entropy* env) {
    // Always pass
    (void)env;
}

void scripted_order_medium(Entropy* env) {
    // Random valid slide
    int pieces[NUM_CELLS];
    int n_pieces = 0;
    for (int i = 0; i < NUM_CELLS; i++) {
        if (env->board[i] != EMPTY_CELL) {
            pieces[n_pieces++] = i;
        }
    }
    if (n_pieces == 0) return;

    // Shuffle and try
    for (int attempt = 0; attempt < n_pieces * 4; attempt++) {
        int cell = pieces[entropy_rand(env) % n_pieces];
        int dir = entropy_rand(env) % 4;
        if (execute_order_slide(env, cell, dir)) return;
    }
}

void scripted_order_hard(Entropy* env) {
    // Greedy: pick the slide that maximizes score gain
    int best_cell = -1;
    int best_dir = -1;
    int best_delta = 0;

    for (int cell = 0; cell < NUM_CELLS; cell++) {
        if (env->board[cell] == EMPTY_CELL) continue;

        int old_score_src = score_affected_lines(env, cell);

        for (int dir = 0; dir < 4; dir++) {
            int dest = compute_slide_destination(env, cell, dir);
            if (dest == cell) continue;

            // Temporarily make the move
            int color = env->board[cell];
            env->board[cell] = EMPTY_CELL;
            env->board[dest] = color;

            int new_score_src = score_affected_lines(env, cell);
            int new_score_dst = score_affected_lines(env, dest);

            // Undo
            env->board[dest] = EMPTY_CELL;
            env->board[cell] = color;

            int old_score_dst = score_affected_lines(env, dest);
            int delta = (new_score_src + new_score_dst) - (old_score_src + old_score_dst);

            if (delta > best_delta) {
                best_delta = delta;
                best_cell = cell;
                best_dir = dir;
            }
        }
    }

    if (best_cell >= 0) {
        execute_order_slide(env, best_cell, best_dir);
    }
}

// ==================== Add Log ====================

void add_log(Entropy* env, float final_score) {
    env->log.episode_length += env->tick;
    env->log.score += final_score;
    env->log.order_score += final_score;
    env->log.episode_return += env->rewards[0];
    env->log.n += 1.0f;
}

// ==================== Reset ====================

void c_reset(Entropy* env) {
    env->tick = 0;
    env->terminals[0] = 0;
    env->turn_number = 0;
    env->current_draw = 0;

    // Clear board
    memset(env->board, 0, sizeof(env->board));

    // Fill bag: 7 of each color
    for (int c = 0; c < NUM_COLORS; c++) {
        env->bag[c] = 7;
    }
    env->total_remaining = NUM_CELLS;

    // Assign role
    if (env->agent_role_config == -1) {
        env->agent_role = (entropy_rand(env) % 2 == 0) ? ROLE_ORDER : ROLE_CHAOS;
    } else {
        env->agent_role = env->agent_role_config;
    }

    // If agent is Order, scripted Chaos makes the first placement
    if (env->agent_role == ROLE_ORDER) {
        if (env->difficulty >= DIFF_HARD) {
            scripted_chaos_hard(env);
        } else {
            scripted_chaos_easy(env);
        }
    } else {
        // Agent is Chaos: draw a piece for the agent
        draw_piece(env);
    }

    compute_observations(env);
}

// ==================== Step ====================

void c_step(Entropy* env) {
    env->tick++;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    int action = env->actions[0];

    if (env->agent_role == ROLE_ORDER) {
        // Agent is Order: decode Order slide or pass
        if (action == ORDER_PASS_ACTION) {
            // Pass - small penalty to discourage always-passing
            env->rewards[0] = -0.1f;
        } else if (action >= ORDER_ACTION_BASE && action < ORDER_PASS_ACTION) {
            int order_action = action - ORDER_ACTION_BASE;
            int cell = order_action / 4;
            int dir = order_action % 4;

            // Score-delta reward shaping: snapshot before slide
            int dest = compute_slide_destination(env, cell, dir);
            int score_before = score_affected_lines(env, cell);
            if (dest != cell) {
                score_before += score_affected_lines(env, dest);
            }

            if (!execute_order_slide(env, cell, dir)) {
                env->rewards[0] = env->reward_invalid;
            } else {
                // Slide succeeded: compute score delta
                // Order wants HIGH score, so positive delta = good
                int score_after = score_affected_lines(env, cell)
                                + score_affected_lines(env, dest);
                int delta = score_after - score_before;
                env->rewards[0] += delta * env->reward_palindrome_delta;
            }
        } else {
            // Invalid action (e.g., Chaos action during Order phase)
            env->rewards[0] = env->reward_invalid;
        }

        // Check if board is full (game over)
        if (env->turn_number >= NUM_CELLS) {
            int final_score = compute_total_score(env);
            float normalized = ((float)final_score - 40.0f) / 30.0f;
            if (normalized > 1.0f) normalized = 1.0f;
            if (normalized < -1.0f) normalized = -1.0f;
            env->rewards[0] = normalized;
            add_log(env, (float)final_score);
            c_reset(env);
            env->terminals[0] = 1;  // Set AFTER reset so Python sees it
            return;
        }

        // Scripted Chaos places next piece
        if (env->difficulty >= DIFF_HARD) {
            scripted_chaos_hard(env);
        } else {
            scripted_chaos_easy(env);
        }

        // Check again after Chaos placed
        if (env->turn_number >= NUM_CELLS) {
            int final_score = compute_total_score(env);
            float normalized = ((float)final_score - 40.0f) / 30.0f;
            if (normalized > 1.0f) normalized = 1.0f;
            if (normalized < -1.0f) normalized = -1.0f;
            env->rewards[0] = normalized;
            add_log(env, (float)final_score);
            c_reset(env);
            env->terminals[0] = 1;
            return;
        }

    } else {
        // Agent is Chaos: decode Chaos placement
        if (action >= CHAOS_ACTION_BASE && action < ORDER_ACTION_BASE) {
            int cell = action;
            if (cell < 0 || cell >= NUM_CELLS || env->board[cell] != EMPTY_CELL
                    || env->current_draw == 0) {
                env->rewards[0] = env->reward_invalid;
            } else {
                // Score-delta reward shaping: snapshot before placement
                int score_before = score_affected_lines(env, cell);

                env->board[cell] = env->current_draw;
                env->turn_number++;

                // Chaos wants LOW score, so negative delta = good
                int score_after = score_affected_lines(env, cell);
                int delta = score_after - score_before;
                env->rewards[0] += -delta * env->reward_palindrome_delta;
            }
        } else {
            // Invalid action (e.g., Order action during Chaos phase)
            env->rewards[0] = env->reward_invalid;
        }

        // Check if board is full
        if (env->turn_number >= NUM_CELLS) {
            int final_score = compute_total_score(env);
            // Chaos wants LOW score, so invert reward
            float normalized = (40.0f - (float)final_score) / 30.0f;
            if (normalized > 1.0f) normalized = 1.0f;
            if (normalized < -1.0f) normalized = -1.0f;
            env->rewards[0] = normalized;
            add_log(env, (float)final_score);
            c_reset(env);
            env->terminals[0] = 1;
            return;
        }

        // Scripted Order slides
        if (env->difficulty >= DIFF_HARD) {
            scripted_order_hard(env);
        } else if (env->difficulty >= DIFF_MEDIUM) {
            scripted_order_medium(env);
        } else {
            scripted_order_easy(env);
        }

        // Draw next piece for agent
        draw_piece(env);
    }

    compute_observations(env);
}

// ==================== Rendering ====================

static const Color ENTROPY_COLORS[8] = {
    {40, 40, 40, 255},       // 0: empty (dark gray)
    {220, 50, 50, 255},      // 1: red
    {50, 120, 220, 255},     // 2: blue
    {50, 180, 50, 255},      // 3: green
    {220, 200, 50, 255},     // 4: yellow
    {160, 50, 200, 255},     // 5: purple
    {220, 130, 30, 255},     // 6: orange
    {50, 200, 200, 255},     // 7: cyan
};

static const Color PUFF_BG = {6, 24, 24, 255};
static const Color PUFF_BG2 = {18, 72, 72, 255};
static const Color PUFF_WHITE = {241, 241, 241, 255};

struct Client {
    float width;
    float height;
};

Client* make_entropy_client(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;
    InitWindow(width, height, "PufferLib Entropy (Hyle 7)");
    SetTargetFPS(60);
    return client;
}

void c_render(Entropy* env) {
    if (env->client == NULL) {
        env->client = make_entropy_client(env->width, env->height);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BG);

    int cell_size = 64;
    int padding = 32;
    int board_px = BOARD_SIZE * cell_size;

    // Draw board background
    DrawRectangle(padding, padding, board_px, board_px, PUFF_BG2);

    // Draw grid and pieces
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            int x = padding + c * cell_size;
            int y = padding + r * cell_size;
            int cell_val = env->board[r * BOARD_SIZE + c];

            // Draw cell border
            DrawRectangleLines(x, y, cell_size, cell_size, PUFF_BG);

            if (cell_val > 0 && cell_val <= NUM_COLORS) {
                // Draw colored circle
                int cx = x + cell_size / 2;
                int cy = y + cell_size / 2;
                int radius = cell_size / 2 - 4;
                DrawCircle(cx, cy, radius, ENTROPY_COLORS[cell_val]);
            }
        }
    }

    // Sidebar info
    int sidebar_x = padding + board_px + 20;
    int text_y = padding;

    const char* role_str = (env->agent_role == ROLE_ORDER) ? "ORDER" : "CHAOS";
    DrawText(TextFormat("Agent Role: %s", role_str), sidebar_x, text_y, 18, PUFF_WHITE);
    text_y += 30;

    DrawText(TextFormat("Turn: %d / 49", env->turn_number), sidebar_x, text_y, 18, PUFF_WHITE);
    text_y += 30;

    if (env->current_draw > 0) {
        DrawText("Current Draw:", sidebar_x, text_y, 18, PUFF_WHITE);
        DrawCircle(sidebar_x + 140, text_y + 9, 12, ENTROPY_COLORS[env->current_draw]);
        text_y += 30;
    }

    DrawText("Bag:", sidebar_x, text_y, 18, PUFF_WHITE);
    text_y += 24;
    for (int c = 0; c < NUM_COLORS; c++) {
        DrawCircle(sidebar_x + 10, text_y + 8, 8, ENTROPY_COLORS[c + 1]);
        DrawText(TextFormat("x%d", env->bag[c]), sidebar_x + 25, text_y, 16, PUFF_WHITE);
        text_y += 22;
    }

    text_y += 10;
    int partial_score = compute_total_score(env);
    DrawText(TextFormat("Score: %d", partial_score), sidebar_x, text_y, 18, PUFF_WHITE);

    EndDrawing();
}

void c_close(Entropy* env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

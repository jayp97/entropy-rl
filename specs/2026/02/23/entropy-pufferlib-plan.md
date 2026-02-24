# Entropy (Hyle 7) — Superhuman Bot via PufferLib

**Date**: 2026-02-23
**Goal**: Build a superhuman reinforcement learning agent for the board game Entropy (Hyle 7) using PufferLib's high-performance Ocean environment framework.

---

## 1. Game Overview

Entropy is an asymmetric two-player abstract strategy board game designed by **Eric Solomon** in 1977. Originally played on a 5x5 board, the modern version (Hyle 7, released 2000) uses a 7x7 board for deeper strategic play. It was awarded 6/6 by Games & Puzzles Magazine in 1981 and has been a fixture at the Mind Sports Olympiad since its inception. It is also the subject of CodeCup 2026, a competitive AI programming contest.

The game's theme is "the eternal conflict in the universe between order and chaos." One player is **Order** (creating palindromic patterns), the other is **Chaos** (preventing them).

---

## 2. Complete Game Rules

### 2.1 Components

- Square board divided into **7x7 cells** (49 cells total)
- **49 counters**: 7 each of 7 different colours
- Small bag to draw counters from

### 2.2 Roles

| Role      | Objective                                              |
| --------- | ------------------------------------------------------ |
| **Order** | Create palindromic colour patterns in rows and columns |
| **Chaos** | Prevent Order from forming patterns                    |

### 2.3 Setup

- The board starts empty
- All 49 counters go into the bag (opaque, drawn unseen)
- Players choose who plays Order and who plays Chaos

### 2.4 Turn Structure

Each turn consists of two phases:

1. **Chaos draws and places**: Chaos draws one counter unseen from the bag and places it on **any** empty square on the board.
2. **Order may slide**: Order may slide **any one counter** on the board (including the one Chaos just placed) **vertically or horizontally** over any number of vacant squares — exactly like a rook in chess. The counter slides in a straight line and stops at the board edge or when it reaches a square adjacent to another counter. Only one counter may occupy a square. **Order may also pass** (make no move).

Turns repeat until all 49 cells are filled.

### 2.5 Scoring

When the board is full, **every horizontal and vertical line** (all 14 lines of 7 cells each) is scored.

**A pattern is any contiguous subsequence of counters that reads identically from either direction** — i.e., a palindrome. A pattern scores **the number of counters in the pattern**. Crucially, **all sub-palindromes within a larger palindrome also score independently**.

**Examples**:

- `red-green-blue-green-red` scores **5** (the full palindrome) **+ 3** (green-blue-green) = **8 total**
- `red-green-red-green-red` scores **5** + 3×3 (three overlapping 3-length palindromes) = **14 total**
- `red-red-red-red` scores **4** + 2×3 + 3×2 = **16 total**

Diagonal lines are **not** scored.

### 2.6 Complete Palindrome Scoring Table

All 30 possible scoring combinations for palindromic patterns of length 2-7. Distinct letters represent distinct colours:

| Pattern | Score |     | Pattern | Score |
| ------- | ----- | --- | ------- | ----- |
| AA      | 2     |     | ABA     | 3     |
| AAA     | 7     |     | ABAAABA | 25    |
| AAAA    | 16    |     | ABAABA  | 18    |
| AAAAA   | 30    |     | ABABA   | 14    |
| AAAAAA  | 50    |     | ABABABA | 27    |
| AAAAAAA | 77    |     | ABACABA | 21    |
| AAABAAA | 29    |     | ABBA    | 6     |
| AABAA   | 12    |     | ABBABBA | 27    |
| AABABAA | 25    |     | ABBBA   | 12    |
| AABBAA  | 16    |     | ABBBBA  | 22    |
| AABBBAA | 23    |     | ABBBBBA | 37    |
| AABCBAA | 19    |     | ABBCBBA | 19    |
|         |       |     | ABCACBA | 15    |
|         |       |     | ABCBA   | 8     |
|         |       |     | ABCBCBA | 21    |
|         |       |     | ABCCBA  | 12    |
|         |       |     | ABCCCBA | 19    |
|         |       |     | ABCDCBA | 15    |

### 2.7 Match Structure

A full match consists of **two rounds**. After the first game, players **swap roles** (Order becomes Chaos and vice versa). The player with the **highest total score across both rounds** wins.

### 2.8 Strategy Notes

- **Average score**: ~75 points
- **Good score**: ~100 points
- **Poor score**: ~50 points
- **Order strategy**: Avoid leaving isolated holes where Chaos can drop awkward colours. The ideal is to maintain no more than two vacant areas. Order achieves this by moving counters to the sides of the board. As the game progresses, Order can calculate the odds of a particular colour coming out of the bag next.
- **Chaos strategy**: Disrupt symmetry formation. Place colours that break existing partial palindromes. Create isolated gaps that force Order into difficult positions.

---

## 3. PufferLib Environment Design

### 3.1 Architecture Overview

Following the established PufferLib Ocean pattern (analyzed from Go, Checkers, Connect4, and Template environments), the implementation consists of 4 files in a new directory:

```
PufferLib/pufferlib/ocean/entropy/
├── entropy.h       # C header: structs, game logic, scoring, AI, rendering
├── entropy.c       # C implementation (minimal — logic lives in .h per convention)
├── entropy.py      # Python PufferEnv wrapper
└── binding.c       # C-Python bridge via env_binding.h
```

Plus configuration and registration:

```
PufferLib/pufferlib/config/ocean/entropy.ini    # Hyperparameters
PufferLib/pufferlib/ocean/environment.py        # Add 'entropy': 'Entropy' to MAKE_FUNCTIONS
PufferLib/pufferlib/ocean/torch.py              # CNN policy class (optional)
```

### 3.2 Game State Representation (C Struct)

```c
typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float order_score;
    float chaos_score;
    float n;
} Log;

typedef struct {
    Log log;
    float* observations;          // float32 observation buffer (402 floats)
    int* actions;                 // single int action per step
    float* rewards;               // single float reward
    unsigned char* terminals;     // episode terminal flag

    // Board state
    int board[49];                // 7x7 board: 0=empty, 1-7=colours
    int bag[7];                   // remaining count of each colour (starts at 7 each)
    int total_remaining;          // pieces left in bag (starts at 49)

    // Turn state
    int phase;                    // 0=Chaos placement, 1=Order slide
    int current_draw;             // colour Chaos just drew (1-7), 0 if Order phase
    int turn_number;              // 0-48, which placement we're on

    // Role tracking
    int agent_role;               // 0=agent plays Order, 1=agent plays Chaos
    int agent_role_config;        // -1=random, 0=Order, 1=Chaos

    // Scoring & rewards
    int final_score;
    float reward_invalid;         // penalty for invalid actions
    float reward_palindrome_delta;// optional per-step shaping

    // Opponent difficulty
    int difficulty;               // 0=random, 1=greedy, 2=hard

    // Rendering (Raylib)
    Client* client;
    int width, height;

    int tick;
} Entropy;
```

### 3.3 Action Space

**`Discrete(246)`** — unified space with phase-dependent interpretation:

| Action Range | Count | Meaning                                                                              |
| ------------ | ----- | ------------------------------------------------------------------------------------ |
| 0–48         | 49    | **Chaos placement**: place drawn counter on cell `i`                                 |
| 49–244       | 196   | **Order slide**: `49 + (cell × 4 + direction)` where direction is 0=N, 1=S, 2=E, 3=W |
| 245          | 1     | **Order pass**: make no move                                                         |

The Order slide moves a piece from `cell` in the specified direction as far as possible (rook-like). The piece stops at the board edge or adjacent to another piece.

Invalid actions (wrong phase, occupied cell, empty source, blocked direction) receive a small negative reward and are ignored.

### 3.4 Observation Space

**`Box(0, 1, shape=(402,), dtype=float32)`**

| Offset  | Size | Description                                                       |
| ------- | ---- | ----------------------------------------------------------------- |
| 0–342   | 343  | 7 colour planes (one-hot): `plane[c][i] = 1.0 if board[i] == c+1` |
| 343–391 | 49   | Empty cell mask: `1.0 if board[i] == 0`                           |
| 392–398 | 7    | Bag state: `bag[c] / 7.0` (normalized remaining count per colour) |
| 399     | 1    | Current draw: `current_draw / 7.0` (0 during Order phase)         |
| 400     | 1    | Phase indicator: `0.0` = Chaos, `1.0` = Order                     |
| 401     | 1    | Turn progress: `turn_number / 49.0`                               |

The one-hot colour planes let the neural network easily distinguish colours. The empty mask directly helps Chaos see valid placements. Bag state is critical for probabilistic reasoning.

### 3.5 Single-Agent Architecture

A **single policy** learns both Order and Chaos roles:

- At episode reset, `agent_role` is randomized (50/50 Order or Chaos)
- The **phase indicator** in observations tells the agent which role is active
- A scripted opponent plays the other role
- This trains a versatile agent that understands both sides of the game

**Turn flow in `c_step`**:

When agent plays **Order**:

1. Scripted Chaos draws and places (happens at end of previous step / at reset)
2. Agent receives observation showing board after Chaos placement
3. Agent's action is decoded as Order slide or pass
4. If game over → compute score, reward, terminal, auto-reset

When agent plays **Chaos**:

1. A piece is drawn from bag, shown in `current_draw`
2. Agent receives observation showing board + drawn piece
3. Agent's action is decoded as Chaos placement
4. Scripted Order slides
5. If game over → compute score, reward, terminal, auto-reset

### 3.6 Palindrome Scoring Algorithm

Efficient O(n³) per line, O(14 × n³) total — trivial at n=7:

```c
int score_line(int line[7]) {
    int total = 0;
    for (int start = 0; start < 7; start++) {
        for (int end = start + 1; end < 7; end++) {
            int len = end - start + 1;
            int is_palindrome = 1;
            for (int k = 0; k < len / 2; k++) {
                if (line[start + k] != line[end - k]) {
                    is_palindrome = 0;
                    break;
                }
            }
            if (is_palindrome) total += len;
        }
    }
    return total;
}

int compute_total_score(Entropy* env) {
    int total = 0;
    // Score all 7 rows
    for (int r = 0; r < 7; r++) {
        int line[7];
        for (int c = 0; c < 7; c++) line[c] = env->board[r * 7 + c];
        total += score_line(line);
    }
    // Score all 7 columns
    for (int c = 0; c < 7; c++) {
        int line[7];
        for (int r = 0; r < 7; r++) line[r] = env->board[r * 7 + c];
        total += score_line(line);
    }
    return total;
}
```

**Validation**: The scoring function must be verified against all 30 entries in the official palindrome table (Section 2.6).

### 3.7 Order Slide Mechanics

```c
int compute_slide_destination(Entropy* env, int cell, int direction) {
    int row = cell / 7, col = cell % 7;
    int dr = 0, dc = 0;
    switch (direction) {
        case 0: dr = -1; break;  // North
        case 1: dr = 1;  break;  // South
        case 2: dc = 1;  break;  // East
        case 3: dc = -1; break;  // West
    }
    int r = row + dr, c = col + dc;
    int last_valid = cell;
    while (r >= 0 && r < 7 && c >= 0 && c < 7) {
        if (env->board[r * 7 + c] != 0) break;
        last_valid = r * 7 + c;
        r += dr; c += dc;
    }
    return last_valid;  // cell itself if no movement possible
}
```

### 3.8 Piece Drawing from Bag

```c
void draw_piece(Entropy* env) {
    if (env->total_remaining <= 0) return;
    int r = rand() % env->total_remaining;
    int cumulative = 0;
    for (int c = 0; c < 7; c++) {
        cumulative += env->bag[c];
        if (r < cumulative) {
            env->current_draw = c + 1;
            env->bag[c]--;
            env->total_remaining--;
            return;
        }
    }
}
```

### 3.9 Scripted Opponents

**Chaos opponents** (used when agent plays Order):

| Difficulty | Strategy                                                                                                                                                  |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Easy (0)   | Random placement on any empty cell                                                                                                                        |
| Hard (1)   | Greedy: for each empty cell, evaluate how much placing there minimizes palindrome potential in affected row + column. Pick the most disruptive placement. |

**Order opponents** (used when agent plays Chaos):

| Difficulty | Strategy                                                                                                                                                                     |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Easy (0)   | Always pass (no slide)                                                                                                                                                       |
| Medium (1) | Random valid slide                                                                                                                                                           |
| Hard (2)   | Greedy: for each possible (piece, direction) slide, compute score delta on affected rows/columns. Pick the slide that maximizes score gain. Pass if no slide improves score. |

### 3.10 Reward Structure

**Terminal reward** (delivered when board is full):

- **Order agent**: `clamp((final_score - 75) / 25, -1, 1)` — positive for above-average, negative for below
- **Chaos agent**: `clamp((75 - final_score) / 25, -1, 1)` — positive for suppressing Order's score

**Shaping rewards** (configurable, default off):

- `reward_invalid`: `-0.1` for invalid actions
- `reward_palindrome_delta`: Optional per-step reward based on score change after Order's slide

### 3.11 CNN Policy Design

The 8-plane 7x7 board representation naturally suits a convolutional architecture:

```python
class Entropy(nn.Module):
    def __init__(self, env, cnn_channels=64, hidden_size=256):
        super().__init__()
        # 8 feature planes (7 colours + empty) as 8-channel 7x7 image
        self.cnn = nn.Sequential(
            nn.Conv2d(8, cnn_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Auxiliary features: bag(7) + draw(1) + phase(1) + progress(1) = 10
        self.aux_encoder = nn.Linear(10, 64)
        self.proj = nn.Sequential(
            nn.Linear(cnn_channels * 49 + 64, hidden_size), nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size, 246)
        self.value_fn = nn.Linear(hidden_size, 1)

    def forward(self, observations):
        batch = observations.shape[0]
        board_planes = observations[:, :392].view(batch, 8, 7, 7)
        aux = observations[:, 392:]
        cnn_out = self.cnn(board_planes)
        aux_out = F.relu(self.aux_encoder(aux))
        hidden = self.proj(torch.cat([cnn_out, aux_out], dim=1))
        return self.actor(hidden), self.value_fn(hidden)
```

### 3.12 Python Wrapper

```python
class Entropy(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128,
                 difficulty=1, agent_role=-1,
                 reward_invalid=-0.1, reward_palindrome_delta=0.0,
                 buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(402,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(246)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            difficulty=difficulty, agent_role=agent_role,
            reward_invalid=reward_invalid,
            reward_palindrome_delta=reward_palindrome_delta)
```

### 3.13 Binding Layer

```c
#include "entropy.h"
#define Env Entropy
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->difficulty = (int)unpack(kwargs, "difficulty");
    env->agent_role_config = (int)unpack(kwargs, "agent_role");
    env->reward_invalid = unpack(kwargs, "reward_invalid");
    env->reward_palindrome_delta = unpack(kwargs, "reward_palindrome_delta");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "order_score", log->order_score);
    assign_to_dict(dict, "chaos_score", log->chaos_score);
    return 0;
}
```

---

## 4. Implementation Phases

### Phase 1: Core Game Logic

- Board state management (init, reset, place, slide)
- Bag drawing with proper randomization
- Palindrome scoring algorithm
- Score verification against all 30 official patterns

### Phase 2: Agent Step Loop

- Phase-dependent action decoding (Chaos placement vs Order slide)
- Invalid action handling with penalty
- Terminal detection (board full)
- Final score computation and reward delivery
- Auto-reset for continuous training

### Phase 3: Scripted Opponents

- Random Chaos (easy)
- Greedy anti-palindrome Chaos (hard)
- Pass-only Order (easy)
- Greedy palindrome-maximizing Order (hard)

### Phase 4: Python Wrapper & Binding

- Observation/action space definitions
- `binding.c` with parameter unpacking
- `entropy.py` PufferEnv subclass

### Phase 5: Registration & Config

- Add to `environment.py` MAKE_FUNCTIONS
- Create `entropy.ini` with training hyperparameters
- Build system integration (auto-discovered by setup.py glob)

### Phase 6: Rendering

- Raylib 7x7 grid visualization
- Colour-coded cells for each of 7 colours
- Sidebar showing: current draw, bag counts, score estimate
- Last-move highlighting

### Phase 7: Training & Validation

- CNN policy in `torch.py`
- Train against easy opponents, then curriculum to hard
- Validate scoring correctness via unit tests
- Benchmark: target score >100 as Order, <50 as Chaos opponent
- Compare against CodeCup 2026 baselines if available

---

## 5. Key Reference Files

| File                                            | Purpose                                                             |
| ----------------------------------------------- | ------------------------------------------------------------------- |
| `PufferLib/pufferlib/ocean/go/go.h`             | Primary reference — board game with float obs, scoring, scripted AI |
| `PufferLib/pufferlib/ocean/go/go.py`            | Python wrapper pattern for board games                              |
| `PufferLib/pufferlib/ocean/checkers/checkers.h` | Two-player board game with action decoding                          |
| `PufferLib/pufferlib/ocean/template/`           | Minimal starter template (4 files)                                  |
| `PufferLib/pufferlib/ocean/env_binding.h`       | Shared C-Python bridge (vec_init, vec_step, unpack)                 |
| `PufferLib/pufferlib/ocean/environment.py`      | Registration dict (MAKE_FUNCTIONS)                                  |
| `PufferLib/pufferlib/ocean/torch.py`            | Custom policy classes                                               |
| `PufferLib/pufferlib/config/ocean/`             | INI config files per environment                                    |

---

## 6. Sources

- [Entropy (board game) — Wikipedia](<https://en.wikipedia.org/wiki/Entropy_(board_game)>)
- [CodeCup 2026 — Rules of Entropy](https://www.codecup.nl/entropy/rules.php)
- [Entropy: An Abstract Game of Perfect Balance — Real Sheldon](https://realsheldon.com/2025/05/13/entropy-board-game/)
- [Entropy Game Rules — mozai.com](https://mozai.com/writing/house_rules/entropy.txt)
- [Official Hyle 7 Rules PDF — tesera.ru](https://tesera.ru/images/items/150398/Hyle%207.PDF)
- [Board Game Guys — Hyle (1979)](https://boardgameguys.com/entropy/)
- [Mind Sports Olympiad — Entropy WC](https://mindsportsolympiad.com/product/entropy-wc/)

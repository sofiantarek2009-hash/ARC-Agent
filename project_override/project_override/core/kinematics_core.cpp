#include "kinematics_core.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

struct EngineSnapshot {
    std::vector<int32_t> canvas;
    std::vector<float> mass_grid;
    int player_r;
    int player_c;
    int facing_r;
    int facing_c;
    float player_push_strength;
    int score;
    int inventory_count;
};

class KinematicsEngine {
public:
    int width;
    int height;
    std::vector<int32_t> canvas;
    std::vector<float> mass_grid;
    int player_r;
    int player_c;
    int facing_r;
    int facing_c;
    float player_push_strength;
    int score;
    int inventory_count;

    std::vector<EngineSnapshot> history_stack;
    EngineSnapshot root_checkpoint;

    KinematicsEngine(int w, int h)
        : width(w), height(h),
          canvas(w * h, CELL_EMPTY),
          mass_grid(w * h, 0.0f),
          player_r(0), player_c(0),
          facing_r(-1), facing_c(0),
          player_push_strength(1.0f),
          score(0), inventory_count(0) {
        commit_root();
    }

    inline int idx(int r, int c) const {
        return r * width + c;
    }

    inline bool in_bounds(int r, int c) const {
        return r >= 0 && r < height && c >= 0 && c < width;
    }

    void set_cell(int r, int c, int val, float mass) {
        if (!in_bounds(r, c)) return;
        int i = idx(r, c);
        canvas[i] = val;
        mass_grid[i] = mass;
        if (val == CELL_PLAYER) {
            player_r = r;
            player_c = c;
        }
    }

    int get_cell(int r, int c) const {
        if (!in_bounds(r, c)) return CELL_WALL;
        return canvas[idx(r, c)];
    }

    float get_mass(int r, int c) const {
        if (!in_bounds(r, c)) return 1e9f;
        return mass_grid[idx(r, c)];
    }

    void commit_root() {
        root_checkpoint = create_snapshot();
    }

    EngineSnapshot create_snapshot() const {
        EngineSnapshot s;
        s.canvas = canvas;
        s.mass_grid = mass_grid;
        s.player_r = player_r;
        s.player_c = player_c;
        s.facing_r = facing_r;
        s.facing_c = facing_c;
        s.player_push_strength = player_push_strength;
        s.score = score;
        s.inventory_count = inventory_count;
        return s;
    }

    void restore_snapshot(const EngineSnapshot& s) {
        canvas = s.canvas;
        mass_grid = s.mass_grid;
        player_r = s.player_r;
        player_c = s.player_c;
        facing_r = s.facing_r;
        facing_c = s.facing_c;
        player_push_strength = s.player_push_strength;
        score = s.score;
        inventory_count = s.inventory_count;
    }

    bool undo() {
        if (history_stack.empty()) {
            return false;
        }
        EngineSnapshot prev = history_stack.back();
        history_stack.pop_back();
        restore_snapshot(prev);
        return true;
    }

    void reset() {
        history_stack.clear();
        restore_snapshot(root_checkpoint);
    }

    bool step_move(int dr, int dc) {
        facing_r = dr;
        facing_c = dc;

        int next_r = player_r + dr;
        int next_c = player_c + dc;

        if (!in_bounds(next_r, next_c)) {
            return false; // Out of bounds static boundary
        }

        int target_val = get_cell(next_r, next_c);

        if (target_val == CELL_WALL || target_val == CELL_DOOR) {
            return false; // Static obstacle
        }

        if (target_val == CELL_EMPTY) {
            // Free move
            set_cell(player_r, player_c, CELL_EMPTY, 0.0f);
            set_cell(next_r, next_c, CELL_PLAYER, 1.0f);
            player_r = next_r;
            player_c = next_c;
            return true;
        }

        if (target_val == CELL_BOX) {
            // Mass-proportional Sokoban pushing
            // Traverse chain of boxes
            std::vector<int> chain_r;
            std::vector<int> chain_c;
            float total_mass = 0.0f;

            int cur_r = next_r;
            int cur_c = next_c;
            while (in_bounds(cur_r, cur_c) && get_cell(cur_r, cur_c) == CELL_BOX) {
                chain_r.push_back(cur_r);
                chain_c.push_back(cur_c);
                total_mass += get_mass(cur_r, cur_c);
                cur_r += dr;
                cur_c += dc;
            }

            // Check cell after chain
            if (!in_bounds(cur_r, cur_c)) {
                return false; // Pushing off grid
            }
            int dest_val = get_cell(cur_r, cur_c);
            if (dest_val != CELL_EMPTY && dest_val != CELL_TARGET) {
                return false; // Blocked by wall or obstacle
            }

            // Mass check: pusher strength vs chain total mass
            if (total_mass > player_push_strength) {
                return false; // Mass resistance exceeds pusher strength
            }

            // Shift chain from back to front
            for (int i = (int)chain_r.size() - 1; i >= 0; --i) {
                int from_r = chain_r[i];
                int from_c = chain_c[i];
                int to_r = from_r + dr;
                int to_c = from_c + dc;
                float box_mass = get_mass(from_r, from_c);
                int box_val = get_cell(from_r, from_c);

                set_cell(to_r, to_c, box_val, box_mass);
            }

            // Move player into first box slot
            set_cell(player_r, player_c, CELL_EMPTY, 0.0f);
            set_cell(next_r, next_c, CELL_PLAYER, 1.0f);
            player_r = next_r;
            player_c = next_c;
            return true;
        }

        // Stepping into collectible or target
        if (target_val == CELL_COLLECTIBLE) {
            score += 10;
            inventory_count += 1;
            set_cell(player_r, player_c, CELL_EMPTY, 0.0f);
            set_cell(next_r, next_c, CELL_PLAYER, 1.0f);
            player_r = next_r;
            player_c = next_c;
            return true;
        }

        if (target_val == CELL_TARGET) {
            score += 50;
            set_cell(player_r, player_c, CELL_EMPTY, 0.0f);
            set_cell(next_r, next_c, CELL_PLAYER, 1.0f);
            player_r = next_r;
            player_c = next_c;
            return true;
        }

        return false;
    }

    bool step_interact() {
        // Interact with facing cell
        int target_r = player_r + facing_r;
        int target_c = player_c + facing_c;

        if (!in_bounds(target_r, target_c)) {
            return false;
        }

        int val = get_cell(target_r, target_c);
        if (val == CELL_COLLECTIBLE) {
            score += 10;
            inventory_count += 1;
            set_cell(target_r, target_c, CELL_EMPTY, 0.0f);
            return true;
        }

        if (val == CELL_TOGGLE) {
            // Toggle doors across the grid
            for (int r = 0; r < height; ++r) {
                for (int c = 0; c < width; ++c) {
                    if (get_cell(r, c) == CELL_DOOR) {
                        set_cell(r, c, CELL_EMPTY, 0.0f);
                    }
                }
            }
            score += 5;
            return true;
        }

        if (val == CELL_TARGET) {
            score += 50;
            return true;
        }

        return false;
    }

    bool step_target(int tr, int tc) {
        if (!in_bounds(tr, tc)) {
            return false;
        }
        int val = get_cell(tr, tc);
        if (val == CELL_TARGET || val == CELL_COLLECTIBLE) {
            score += 25;
            if (val == CELL_COLLECTIBLE) {
                inventory_count += 1;
                set_cell(tr, tc, CELL_EMPTY, 0.0f);
            }
            return true;
        }
        return true;
    }

    bool step(int action, int tr, int tc) {
        if (action == ACTION7_UNDO) {
            return undo();
        }
        if (action == ACTION_RESET) {
            reset();
            return true;
        }

        // Push state before action
        history_stack.push_back(create_snapshot());

        bool ok = false;
        switch (action) {
            case ACTION1_UP:
                ok = step_move(-1, 0);
                break;
            case ACTION2_DOWN:
                ok = step_move(1, 0);
                break;
            case ACTION3_LEFT:
                ok = step_move(0, -1);
                break;
            case ACTION4_RIGHT:
                ok = step_move(0, 1);
                break;
            case ACTION5_INTERACT:
                ok = step_interact();
                break;
            case ACTION6_TARGET:
                ok = step_target(tr, tc);
                break;
            default:
                ok = false;
                break;
        }

        if (!ok) {
            // Discard snapshot on failed action so undo doesn't get stuck
            history_stack.pop_back();
        }

        return ok;
    }
};

// =============================================================================
// C API Implementations
// =============================================================================

extern "C" {

KinematicsEngineHandle create_kinematics_engine(int width, int height) {
    return (KinematicsEngineHandle)(new KinematicsEngine(width, height));
}

void destroy_kinematics_engine(KinematicsEngineHandle handle) {
    if (handle) {
        delete (KinematicsEngine*)handle;
    }
}

void set_grid_cell(KinematicsEngineHandle handle, int r, int c, int val, float mass) {
    if (!handle) return;
    ((KinematicsEngine*)handle)->set_cell(r, c, val, mass);
}

int get_grid_cell(KinematicsEngineHandle handle, int r, int c) {
    if (!handle) return CELL_WALL;
    return ((KinematicsEngine*)handle)->get_cell(r, c);
}

float get_cell_mass(KinematicsEngineHandle handle, int r, int c) {
    if (!handle) return 1e9f;
    return ((KinematicsEngine*)handle)->get_mass(r, c);
}

void get_grid_canvas(KinematicsEngineHandle handle, int32_t* out_buffer) {
    if (!handle || !out_buffer) return;
    KinematicsEngine* eng = (KinematicsEngine*)handle;
    std::memcpy(out_buffer, eng->canvas.data(), eng->canvas.size() * sizeof(int32_t));
}

void set_player_state(KinematicsEngineHandle handle, int r, int c, float push_strength) {
    if (!handle) return;
    KinematicsEngine* eng = (KinematicsEngine*)handle;
    eng->player_r = r;
    eng->player_c = c;
    eng->player_push_strength = push_strength;
    eng->set_cell(r, c, CELL_PLAYER, 1.0f);
}

void get_player_state(KinematicsEngineHandle handle, int* out_r, int* out_c, float* out_strength) {
    if (!handle) return;
    KinematicsEngine* eng = (KinematicsEngine*)handle;
    if (out_r) *out_r = eng->player_r;
    if (out_c) *out_c = eng->player_c;
    if (out_strength) *out_strength = eng->player_push_strength;
}

int step_action(KinematicsEngineHandle handle, int action_code, int target_r, int target_c) {
    if (!handle) return 0;
    return ((KinematicsEngine*)handle)->step(action_code, target_r, target_c) ? 1 : 0;
}

int undo_step(KinematicsEngineHandle handle) {
    if (!handle) return 0;
    return ((KinematicsEngine*)handle)->undo() ? 1 : 0;
}

void reset_to_root(KinematicsEngineHandle handle) {
    if (!handle) return;
    ((KinematicsEngine*)handle)->reset();
}

void commit_root_checkpoint(KinematicsEngineHandle handle) {
    if (!handle) return;
    ((KinematicsEngine*)handle)->commit_root();
}

int get_history_depth(KinematicsEngineHandle handle) {
    if (!handle) return 0;
    return (int)((KinematicsEngine*)handle)->history_stack.size();
}

int compute_deterministic_intercept_c(
    int source_r, int source_c,
    int target_r, int target_c,
    int velocity_r, int velocity_c,
    int speed,
    int metric,
    int max_time,
    int* out_intercept_r,
    int* out_intercept_c,
    int* out_intercept_time
) {
    if (speed <= 0) speed = 1;
    if (max_time <= 0) max_time = 128;

    for (int t = 0; t <= max_time; ++t) {
        int tr = target_r + t * velocity_r;
        int tc = target_c + t * velocity_c;

        int dr = std::abs(tr - source_r);
        int dc = std::abs(tc - source_c);

        int dist = 0;
        if (metric == 1) {
            // Manhattan L1
            dist = dr + dc;
        } else if (metric == 2) {
            // Chebyshev L_inf
            dist = std::max(dr, dc);
        } else {
            // Euclidean L2
            dist = (int)std::ceil(std::sqrt((double)(dr * dr + dc * dc)));
        }

        if (dist <= speed * t) {
            if (out_intercept_r) *out_intercept_r = tr;
            if (out_intercept_c) *out_intercept_c = tc;
            if (out_intercept_time) *out_intercept_time = t;
            return 1;
        }
    }

    // Fallback: final position at max_time
    if (out_intercept_r) *out_intercept_r = target_r + max_time * velocity_r;
    if (out_intercept_c) *out_intercept_c = target_c + max_time * velocity_c;
    if (out_intercept_time) *out_intercept_time = max_time;
    return 0;
}

}

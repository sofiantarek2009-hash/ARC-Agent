#ifndef KINEMATICS_CORE_H
#define KINEMATICS_CORE_H

#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Action definitions
enum ActionType {
    ACTION_NONE = 0,
    ACTION1_UP = 1,
    ACTION2_DOWN = 2,
    ACTION3_LEFT = 3,
    ACTION4_RIGHT = 4,
    ACTION5_INTERACT = 5,
    ACTION6_TARGET = 6,
    ACTION7_UNDO = 7,
    ACTION_RESET = 8
};

// Cell type definitions
enum CellType {
    CELL_EMPTY = 0,
    CELL_WALL = 1,
    CELL_PLAYER = 2,
    CELL_BOX = 3,
    CELL_TARGET = 4,
    CELL_COLLECTIBLE = 5,
    CELL_DOOR = 6,
    CELL_TOGGLE = 7
};

// C-compatible opaque handle
typedef void* KinematicsEngineHandle;

// Engine lifecycle
KinematicsEngineHandle create_kinematics_engine(int width, int height);
void destroy_kinematics_engine(KinematicsEngineHandle handle);

// Grid manipulation
void set_grid_cell(KinematicsEngineHandle handle, int r, int c, int val, float mass);
int get_grid_cell(KinematicsEngineHandle handle, int r, int c);
float get_cell_mass(KinematicsEngineHandle handle, int r, int c);
void get_grid_canvas(KinematicsEngineHandle handle, int32_t* out_buffer);

// Player setup
void set_player_state(KinematicsEngineHandle handle, int r, int c, float push_strength);
void get_player_state(KinematicsEngineHandle handle, int* out_r, int* out_c, float* out_strength);

// Action execution
// Returns 1 on success, 0 on failure/collision
int step_action(KinematicsEngineHandle handle, int action_code, int target_r, int target_c);

// History mechanics (Undo and Reset)
int undo_step(KinematicsEngineHandle handle);
void reset_to_root(KinematicsEngineHandle handle);
void commit_root_checkpoint(KinematicsEngineHandle handle);
int get_history_depth(KinematicsEngineHandle handle);

// Deterministic Intercept Geometry
// Solves min integer t >= 0: dist((sr, sc), Target(t)) <= speed * t
// Target(t) = (tr + t * vr, tc + t * vc)
// Metric: 1 = Manhattan (L1), 2 = Chebyshev (L_inf), 3 = Euclidean (L2)
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
);

#ifdef __cplusplus
}
#endif

#endif // KINEMATICS_CORE_H

# Codex Soccer 3D RL Environment — Build Plan

This document outlines the step-by-step build plan for creating the **CodexSoccer** Unity project. It captures the scene composition, prefabs, and system architecture required to integrate a custom reinforcement learning (RL) loop without implementing the gameplay scripts yet. Use this plan as a checklist while building the project.

---

## Phase 0 — Project Setup (10–15 minutes)
1. Create a new Unity 3D project named **CodexSoccer** (3D Core or URP template).
2. Configure **Project Settings → Time**:
   - Fixed Timestep: **0.02** (50 Hz).
   - Maximum Allowed Timestep: **0.1**.
3. Configure **Project Settings → Physics**:
   - Gravity: **(0, -9.81, 0)**.
4. Add the following layers: `PlayerA`, `PlayerB`, `Ball`, `Field`, `Goals`.
5. Create project folders: `Assets/Scenes`, `Assets/Prefabs`, `Assets/Materials`, `Assets/Scripts`, `Assets/ScriptableObjects`.

---

## Phase 1 — Scene and Prefabs (Core World)
1. Create and save `Scenes/SoccerScene`.
2. **Field**
   - Add a Plane named `Field`, scale to **(3, 1, 2)** to approximate 60 m × 40 m.
   - Assign the `Field` layer.
   - Add four thin Cube walls (colliders enabled) around the perimeter, height ≈ 2.5.
3. **Goals**
   - Create empty objects `GoalA` at **(-28, 0, 0)** and `GoalB` at **(28, 0, 0)**.
   - Add child trigger cubes (`GoalA_Trigger`, `GoalB_Trigger`) scaled to **(1, 2.5, 7)**, set **Is Trigger** to true, assign the `Goals` layer.
4. **Ball Prefab**
   - Create a Sphere named `Ball`, scale **(0.35, 0.35, 0.35)**, layer `Ball`.
   - Add Rigidbody with: Mass **0.43**, Drag **0.05**, Angular Drag **0.05**, Interpolate **Interpolate**, Collision Detection **Continuous**.
   - Enable Gravity.
   - Create `Materials/BallPhysic` Physic Material: Dynamic Friction **0.2**, Static Friction **0.2**, Bounciness **0.35**, Combine modes **Multiply**.
   - Assign the physic material to the collider and save the object to `Prefabs/Ball.prefab`.
5. **Player Prefab**
   - Create a Capsule named `Player`, assign Rigidbody with: Mass **70**, Drag **0**, Angular Drag **0.05**, Interpolate **Interpolate**, Collision Detection **Continuous Dynamic**.
   - Enable Gravity.
   - Add a child empty object `Foot` at local position ≈ **(0, 0.4, 0.5)**.
   - Save as `Prefabs/Player.prefab`.
6. **Scene Placement**
   - Instantiate `Player` twice as `AgentA` and `AgentB` at **(-4, 0.5, 0)** and **(4, 0.5, 0)** respectively.
   - Assign layers `PlayerA` and `PlayerB` to the instances.
   - Instantiate `Ball` at **(0, 0.35, 0)**.
   - Confirm walls and field colliders prevent objects from leaving the play area.

---

## Phase 2 — Cameras, Lighting, and Materials
1. Configure the Main Camera:
   - Option A (top-down): Position **(0, 25, 0)**, Rotation **(90, 0, 0)**.
   - Option B (isometric): Position **(0, 18, -22)**, Rotation **(35, 0, 0)**.
2. Add a Directional Light with soft shadows for consistent lighting.
3. Create materials for rapid visual debugging: `FieldMat` (green), `GoalMat` (white), `PlayerAMat` (blue), `PlayerBMat` (red), `BallMat` (white/black) and assign them to the respective objects.

---

## Phase 3 — Managers and Responsibilities (No Code Yet)
Add empty scripts later, but prepare scene objects and references now:

- **EnvironmentManager** (root-level GameObject)
  - Holds references to `AgentA`, `AgentB`, `Ball`, `GoalA_Trigger`, `GoalB_Trigger`, field bounds.
  - Manages resets, goal detection, episode timing, and foul/out-of-bounds handling.
- **MatchManager** (sibling object)
  - Tracks score, match clock, kickoff logic, and possession.
- **BallController** (attach to Ball prefab)
  - Will expose reset, kick impulse, drag tuning, and optional speed clamping.
- **AgentController** (attach to Player prefab)
  - Processes action vector `[move_x, move_z, kick]`, applies forces, manages ball interaction via `Foot`, and exposes observation getters.
- **GoalTrigger** (on trigger cubes)
  - Notifies managers of goal events and prevents rapid repeat triggers until reset.

Assign public fields in the Inspector so the references are ready once scripts are implemented.

---

## Phase 4 — Observation and Action Contracts
Define the data structures for RL integration:

- **Action Vector** (per agent): `move_x ∈ [-1, 1]`, `move_z ∈ [-1, 1]`, `kick ∈ {0, 1}` (optionally continuous).
- **Observation Struct** (per step):
  - Agent A: position and velocity.
  - Agent B: position and velocity.
  - Ball: position and velocity.
  - Goal positions (optional because static).
  - Optional engineered features: distances and angles to the ball, opponent, and goals.
- **Termination Conditions**: goal scored, max step limit reached, ball out of bounds, or stalemate timer expiration.
- **Reward Concept**: +1 for scoring, −1 for conceding, plus potential shaping rewards (ball velocity toward goal, possession, spacing).

---

## Phase 5 — Physics Tuning and Gameplay Feel
1. Configure physic materials: low friction for the field, slightly higher friction for players.
2. Target player speed around 3–5 m/s; clamp Rigidbody velocity if needed.
3. Tune kick impulses so shots from midfield can reach the goal without excessive airtime.
4. Use `FixedUpdate` for all gameplay physics to match the 50 Hz timestep.
5. Enable Rigidbody interpolation for smooth visuals while maintaining deterministic physics.

---

## Phase 6 — Determinism and Reproducibility
1. Drive gameplay from `FixedUpdate()` and the fixed timestep configuration.
2. Centralize random number generation for spawn sides and kickoff noise; store seeds per episode.
3. On reset, zero all velocities, reposition agents and ball slightly above the ground (y ≈ 0.35–0.5), and add a short delay (~0.2 s) to allow objects to settle.

---

## Phase 7 — RL I/O Integration (Future Work)
Consider communication methods between Unity and the external RL trainer:

- TCP socket (Unity server on port such as 5555).
- Named pipes (Windows) for low-latency local training.
- gRPC or WebSocket for structured messaging.
- Unity ML-Agents toolkit for built-in training support.

Use a simple JSON protocol initially:
- **Unity → RL**: `{obs, reward, done, info}`.
- **RL → Unity**: `{action: [ax, az, kick]}`.
- Process one action per `FixedUpdate` or use action repetition to match simulation ticks.

---

## Phase 8 — Manual Testing Scenarios
Before training, validate the setup manually:

1. Map temporary WASD/arrow controls to `AgentA` with spacebar to kick.
2. Leave `AgentB` idle to confirm collisions and goal detection.
3. Script a simple chasing behavior for `AgentB` as a smoke test.
4. Test edge cases: ball stuck in corners, high-speed impacts, consecutive goals, out-of-bounds resets.

---

## Phase 9 — Scaling Beyond Two Agents
Once the two-agent MVP is stable:

- Add teammates/opponents for 4v4 play.
- Introduce roles (striker, defender) and stamina/sprint systems.
- Detect possession (last contact within ~0.5 s).
- Implement fouls, tackles, and restarts (throw-ins, goal-kicks) for realism.
- Replace capsules with rigged humanoids and animation controllers.
- Develop curricula: small field to large, static ball to moving starts, sparse to shaped rewards.

---

## Milestone Checklist
- [ ] Scene `SoccerScene` with field, walls, players, ball, and goal triggers.
- [ ] Prefabs: `Player`, `Ball`, optional `Field`, goal trigger cubes.
- [ ] Manager objects placed with inspector references.
- [ ] Layers and physics settings configured.
- [ ] Camera and lighting finalized for top-down view.
- [ ] Observation/action/reset contracts documented for RL integration.

---

## Tips and Gotchas
- Enable **Gizmos** to visualize triggers and spawn points during setup.
- Spawn agents and ball slightly above the plane to avoid collider overlap.
- Maintain consistent units (1 Unity unit = 1 meter).
- Start with simple capsules and collisions before adding animation complexity.


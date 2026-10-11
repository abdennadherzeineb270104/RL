"""
train/train_ppo.py  —  Bin Packing 2D (CPU + RAM)
==================================================
Modifications par rapport à la version précédente :
  - cpu_per_node=8, ram_per_node=16  → capacité des nœuds
  - zeta=0.1                         → pénalité fragmentation 2D
  - obs_dim = 10×8 + 13 = 93
"""

import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env.scheduling_env import SchedulingEnv

os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    USE_MASKING = True
    print("✅ MaskablePPO activé (masquage d'actions réel)")
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    USE_MASKING = False
    print("⚠️  sb3-contrib non trouvé → PPO standard")

# ── Hyperparamètres ───────────────────────────────────────────────────────────
GAMMA = 0.5
ALPHA = 0.1
BETA  = 0.05
DELTA = 0.2
ZETA  = 0.1   # pénalité fragmentation 2D ← nouveau

# Capacité des nœuds (bin packing 2D)
CPU_PER_NODE = 8.0    # cœurs
RAM_PER_NODE = 16.0   # GB


def linear_schedule(initial_lr: float):
    def schedule(progress: float) -> float:
        return max(initial_lr * progress, 5e-6)
    return schedule


def make_env_fn():
    env = SchedulingEnv(
        n_nodes=3, window_size=10,
        cpu_per_node=CPU_PER_NODE,
        ram_per_node=RAM_PER_NODE,
        gamma=GAMMA, alpha=ALPHA, beta=BETA, delta=DELTA, zeta=ZETA,
    )
    if USE_MASKING:
        env = ActionMasker(env, lambda e: e.action_masks())
    return env


class EpisodeLogger(BaseCallback):
    def __init__(self, log_every: int = 20):
        super().__init__()
        self.episode_rewards:   list  = []
        self.episode_makespans: list  = []
        self._current_reward:   float = 0.0
        self._best_makespan:    float = float("inf")
        self.log_every = log_every

    def _on_step(self) -> bool:
        self._current_reward += float(self.locals["rewards"][0])

        if self.locals["dones"][0]:
            try:
                ms = float(self.training_env.get_attr("last_makespan")[0])
            except Exception:
                ms = 0.0

            self.episode_rewards.append(self._current_reward)
            self.episode_makespans.append(ms)

            if 0 < ms < self._best_makespan:
                self._best_makespan = ms
                self.model.save("results/ppo_scheduler_best")

            self._current_reward = 0.0
            n = len(self.episode_rewards)

            if n % self.log_every == 0:
                w        = min(self.log_every, n)
                mean_r   = np.mean(self.episode_rewards[-w:])
                valid_ms = [m for m in self.episode_makespans[-w:] if m > 0]
                mean_ms  = np.mean(valid_ms) if valid_ms else 0.0
                best     = self._best_makespan if self._best_makespan < float("inf") else 0.0
                print(
                    f"  [Ep {n:5d}]  "
                    f"Reward={mean_r:7.3f}  "
                    f"Makespan={mean_ms:7.1f}  "
                    f"Best={best:7.1f}"
                )
        return True

    def save(self, reward_path: str, makespan_path: str):
        np.save(reward_path,   np.array(self.episode_rewards,   dtype=np.float32))
        np.save(makespan_path, np.array(self.episode_makespans, dtype=np.float32))
        print(f"✅ {len(self.episode_rewards)} épisodes → {reward_path}")


# ── ENV ───────────────────────────────────────────────────────────────────────
env = DummyVecEnv([make_env_fn])

obs_dim   = env.observation_space.shape[0]
n_actions = env.action_space.n

# ── MODÈLE ────────────────────────────────────────────────────────────────────
model = MaskablePPO(
    policy="MlpPolicy",
    env=env,
    verbose=0,
    n_steps=2048,
    batch_size=256,
    n_epochs=15,
    learning_rate=linear_schedule(3e-4),
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=None,   # désactivé si tensorboard non installé
)

print("=" * 65)
print(f"  PPO LIGO-30  BIN PACKING 2D  |  obs_dim={obs_dim}  actions={n_actions}")
print(f"  Nœuds : {3} × {CPU_PER_NODE:.0f} CPU / {RAM_PER_NODE:.0f}GB RAM")
print(f"  Récompense : γ={GAMMA} α={ALPHA} β={BETA} δ={DELTA} ζ={ZETA}(frag)")
print(f"  Masquage   : {'MaskablePPO (actif)' if USE_MASKING else 'PPO standard'}")
print("=" * 65)

# ── ENTRAÎNEMENT ──────────────────────────────────────────────────────────────
callback = EpisodeLogger(log_every=20)
model.learn(total_timesteps=300_000, callback=callback, progress_bar=False)

model.save("results/ppo_scheduler")
callback.save("results/rewards.npy", "results/train_makespans.npy")
env.close()

# ── RÉSUMÉ ────────────────────────────────────────────────────────────────────
rewards   = np.array(callback.episode_rewards)
makespans = np.array([m for m in callback.episode_makespans if m > 0], dtype=np.float32)
n = len(rewards)
q = max(n // 4, 1)

print("\n" + "=" * 65)
print("  RÉSUMÉ FINAL — Bin Packing 2D")
print("=" * 65)
print(f"  Épisodes             : {n}")
print(f"  Reward  début → fin  : {rewards[:q].mean():.3f} → {rewards[-q:].mean():.3f}  "
      f"{'✅' if rewards[-q:].mean() > rewards[:q].mean() else '⚠️'}")
if len(makespans) > q:
    print(f"  Makespan début → fin : {makespans[:q].mean():.1f} → {makespans[-q:].mean():.1f}  "
          f"{'✅' if makespans[-q:].mean() < makespans[:q].mean() else '⚠️'}")
    print(f"  Best makespan        : {makespans.min():.1f}")
print("=" * 65)

# Seed-42 gate

Run `tools/audit_d2_seed42_gate.py` only after fixed validation reports exist.
All twelve booleans in its input must be true, all numeric endpoints finite,
and all three validation positions present. A passed file permits later config
generation only. It does not authorize training or test evaluation.

The gate blocks multi-seed work when every D2 dose above zero remains below
noisy, when improvement is driven by one frame/position, when teacher/frozen
parameters move, or when fixed-final and preregistered-best conclusions disagree.


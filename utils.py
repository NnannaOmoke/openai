def explore_env(env):
    for _ in range(200):
        env.step(env.action_space.sample())
    return

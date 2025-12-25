liquid particle filters

based on:
- particle filter recurrent networks: https://arxiv.org/abs/1905.12885 + https://github.com/Yusufma03/pfrnns
- neural circuit policies and liquid neural networks: https://arxiv.org/abs/1803.08554 + https://github.com/mlech26l/ncps

intent:
- broad: explicit modelling and quantification of uncertainty
- specific use case in example: RSSI signal processing


how to use my code:

- git clone this repo
- cd into the repo + poetry install

-> multiple architectures approaches to neural particle filtering are available in the nn/ directory:
    - generalist: 
        - State-Level (StateLevelPFCell, PFCfC, PFLTC)
        - Parameter-Level (ParamLevelPFCell, PFCfC, PFLTC)
        - Dual-Level (DualLevelPFCell, PFCfC, PFLTC) (State + Parameter)
        - SDE-Level (SDELevelPFCell, PFCfC, PFLTC)
    - specialist: 
        - Spatial PFNCP (SpatialPFNCP) -> maps latent beleif state to an interpretable probabilistic heatmap

why did i work on this?

currently playing around a system to track rf signal sources.
setup:
    - a sensor with dual yagi antennas is mobile and can be controlled by a rl agent
    - rl agent must track signal source and navigate to it (goal is to reach tracked signal source)
    - currently uses classical particle filter (monte carlo) to estimate signal source position and provide a belief state  prior its decision making step with the intent of providing it with more surface area to work with
    - came around this cool paper on particle filter recurrent networks: https://arxiv.org/abs/1905.12885
    - wanted to try to build a liquid particle filter because i love liquid nets (plz Ramin, hire me :P) they are awesome. everybody should have a liquid net as a pet companion.

thus, examples/RSSI is an example application of a particle filter liquid net for RSSI signal processing:
given a sensory input, the system must estimate the a belief state of the signal source position..

- input: [RSSI_front, RSSI_back, rotation, speed, sensor_heading]
    - input if trigonometric_processing == true: [RSSI_front, RSSI_back, rotation, speed, sensor_heading_sin, sensor_heading_cos] (to avoid 0 -> 1 sharp transition at the 180° angle)
- primary output: [distance_norm, bearing_norm]
    - output if trigonometric_processing == true: [distance_norm, bearing_norm_sin, bearing_norm_cos]
- secondary output: particles (probabilistic hypothesis about the target's relative position):
each particle is projected by the spatial head into interpretable spatial coordinates -> they vote for a specific location of the signal source (hypothesized position + spatial uncertainty + importance weight)

dataset is generated using the RL env setup (not included here, but a built dataset is available in examples/RSSI/data/dpf_dataset_polar_double_single_target): antenna moedling with directional (dual yaggi with csv antenna patterns) and omnidirectional antenna patterns to simulate realistic signal directionality. Uses log-distance path loss models with fading to generate noisy RSSI observations (forcing the filter to learn noise rejection). Different trajectories are used at random for both sensor and target in any single episode.



- run the rssi example:
    - training:
        poetry run python examples/RSSI/train_rssi.py --config examples/RSSI/configs/default.yaml --run_id run_v4 --debug_viz --n_debug_episodes 2 --output_dir examples/RSSI/output/models/run_v4/
    - inference (use existing checkpoints):
        poetry run python examples/RSSI/inference_rssi.py --model_dir examples/RSSI/output/models/run_v3/models/run_v3 --n_episodes 2

-> debug viz of the trained model:

![Debug Episode 1](examples/RSSI/output/models/run_v3/models/run_v3/inference_output/debug_episode_1.gif)

top left plot shows the belief state of the model regarding the target position,
top middle plot shows the egocentric heatmap of the belief state -> could be processed by a cnn prior to rl agent.

tl;dr: that was a cool rabbit hole to go down to

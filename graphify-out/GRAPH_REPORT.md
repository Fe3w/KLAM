# Graph Report - KALM  (2026-09-15)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 5484 nodes · 14238 edges · 222 communities (129 shown, 26 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 1882 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `689b96fe`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pybullet_utils.py
- 2
- i
- a
- _
- fe
- d
- g
- k
- Y
- robot_utils.py
- n
- .update
- Pose
- ge
- .touch
- utils.py
- model.py
- o
- 368
- re
- .defaults
- l
- kn
- jupyterlab-manager/package.json
- te
- 7032
- get_joint_positions
- MIDAS.py
- 543.1c80835729e2b26eb76d.js
- index.js
- add_line
- me
- multiply
- open3d/package.json
- upsamplers.py
- dump_body
- Ae
- T
- plan_nonholonomic_motion
- ee
- y
- get_all_links
- x
- train_jbu_upsampler.py
- a
- Saver
- .addClass
- i
- b
- BaseTrainTester
- models.py
- clone_body
- 84.09c61a35dce1705f.js
- d
- block.py
- PybulletRobot
- .dispose
- featup/util.py
- losses.py
- datasets/util.py
- MaskedAutoencoderViT
- c
- m
- .__init__
- get_length
- get_image
- get_featurizer
- gu
- et
- DINO.py
- dependencies
- o
- 2888
- clip.py
- 160.235911d8e26dc735.js
- train_implicit_upsampler.py
- DINOv2.py
- VGG
- n
- layers/__init__.py
- .__init__
- ResNet
- create_shape
- i
- main_kalm_eval_robot.py
- jupyterlab-plotly/package.json
- KeypointPredictor
- n
- be
- KalmDiffuser
- a
- de
- Qa
- ss
- HighResEmbs.py
- modules/layers.py
- s
- devDependencies
- 3816
- m
- div_round_up
- EmbeddingFile
- ChannelNorm
- scripts
- t
- 9907
- we
- RelPropSimple
- mask_proposal.py
- fu
- je
- e
- ai
- get_region_proposals
- 3018
- 1944
- dt
- trunc_normal_
- s
- r
- r
- Ct
- it
- gf
- es
- extract_tag_content
- VGG_spread
- ne
- qt
- mi
- oo
- or
- SingleImageDataset
- af
- za
- adaptive_conv.cpp
- VisionTransformer
- PatchEmbed
- tc
- 6439
- _i
- jupyterlab
- jupyterlab
- scripts
- devDependencies
- 0.e889754e73c03c32.js
- A
- remoteEntry.c8c9eed2cdefdeac.js
- pygrun
- repository
- 1386
- 5098
- .encode_image
- train_kalmdiffuser.sh

## God Nodes (most connected - your core abstractions)
1. `2()` - 404 edges
2. `_` - 206 edges
3. `o()` - 142 edges
4. `i()` - 127 edges
5. `a()` - 121 edges
6. `t()` - 117 edges
7. `m()` - 116 edges
8. `r()` - 110 edges
9. `n()` - 108 edges
10. `e()` - 108 edges

## Surprising Connections (you probably didn't know these)
- `find_consistent_match_points_on_all_query_images()` --uses--> `ImageRecord`  [INFERRED]
  scripts/main_kalm_distill_keypoints.py → kalm/keypoint_predictor.py
- `save_as_kalmdiffuser_record()` --uses--> `ImageRecord`  [INFERRED]
  scripts/main_kalm_distill_keypoints.py → kalm/keypoint_predictor.py
- `find_consistent_match_points_on_all_query_images()` --uses--> `KeypointPredictor`  [INFERRED]
  scripts/main_kalm_distill_keypoints.py → kalm/keypoint_predictor.py
- `visualize_prediction()` --uses--> `KeypointPredictor`  [INFERRED]
  scripts/main_kalm_eval_robot.py → kalm/keypoint_predictor.py
- `find_consistent_match_points_on_all_query_images()` --uses--> `MatchingAlgoParams`  [INFERRED]
  scripts/main_kalm_distill_keypoints.py → kalm/keypoint_predictor.py

## Import Cycles
- None detected.

## Communities (222 total, 26 thin omitted)

### Community 0 - "pybullet_utils.py"
Cohesion: 0.01
Nodes (212): add_button(), add_parameter(), add_segments(), batch_ray_collision(), cache_decorator(), circular_difference(), circular_interval(), contact_collision() (+204 more)

### Community 1 - "2"
Cohesion: 0.02
Nodes (189): 2(), a2(), c0(), c1(), c4(), c5(), c6(), c7() (+181 more)

### Community 2 - "i"
Cohesion: 0.04
Nodes (148): a(), l(), o(), i(), A(), c(), I(), n() (+140 more)

### Community 3 - "a"
Cohesion: 0.09
Nodes (121): eM(), ez(), a(), eF(), ej(), eR(), n(), b() (+113 more)

### Community 4 - "_"
Cohesion: 0.04
Nodes (75): _, activatePlugin(), activatePlugins(), addAnnotation(), An(), applyPrevMaps(), atrule(), brighter() (+67 more)

### Community 5 - "fe"
Cohesion: 0.02
Nodes (5): fe, mt, p(), ye, Z

### Community 6 - "d"
Cohesion: 0.04
Nodes (6): s(), d(), r(), h(), r(), raw()

### Community 7 - "g"
Cohesion: 0.04
Nodes (9): clearAnnotation(), g(), generate(), Kt(), o(), resolve(), s(), toBase64() (+1 more)

### Community 8 - "k"
Cohesion: 0.04
Nodes (38): gt, j(), k, ce(), de(), ee(), F(), ge() (+30 more)

### Community 9 - "Y"
Cohesion: 0.07
Nodes (65): 204(), b(), E(), M(), n(), P(), S(), v() (+57 more)

### Community 10 - "robot_utils.py"
Cohesion: 0.04
Nodes (34): add_data_path(), create_floor(), create_mesh(), create_obj(), create_sphere(), ensure_dir(), get_base_name(), get_body_info() (+26 more)

### Community 12 - "n"
Cohesion: 0.04
Nodes (3): n(), v(), wt

### Community 13 - ".update"
Cohesion: 0.06
Nodes (4): ci, ei, he(), oe()

### Community 14 - "Pose"
Cohesion: 0.04
Nodes (71): aabb2d_from_aabb(), aabb_contains_aabb(), aabb_contains_point(), aabb_empty(), aabb_from_extent_center(), aabb_intersection(), approximate_as_cylinder(), approximate_as_prism() (+63 more)

### Community 15 - "ge"
Cohesion: 0.04
Nodes (3): ge, hasData(), pe

### Community 16 - ".touch"
Cohesion: 0.04
Nodes (7): At(), copy(), li, ni, ri, Tt(), vt()

### Community 17 - "utils.py"
Cohesion: 0.07
Nodes (55): DINOFeatureExtractor, approx_equal(), calc_grid_cell_tlbr(), center_crop(), draw_grid(), draw_seg_on_im(), draw_text(), extract_fpfh_feature() (+47 more)

### Community 18 - "model.py"
Cohesion: 0.05
Nodes (26): interpolate_positional_embedding(), Tensor, Interpolate the positional encoding for CLIP to the number of patches in the…, AttentionPool2d, Bottleneck, build_model(), CLIP, convert_weights() (+18 more)

### Community 19 - "o"
Cohesion: 0.08
Nodes (62): ex(), s(), ar(), at(), b(), g(), k(), M() (+54 more)

### Community 20 - "368"
Cohesion: 0.05
Nodes (46): 368(), e0(), e8(), e9(), eb(), eC(), eD(), eh() (+38 more)

### Community 22 - ".defaults"
Cohesion: 0.05
Nodes (20): ai, bt(), di, ft(), ht, ii, In, Jt() (+12 more)

### Community 23 - "l"
Cohesion: 0.07
Nodes (7): r(), ct(), E(), ie(), Ke, l(), xe()

### Community 24 - "kn"
Cohesion: 0.06
Nodes (8): darker(), Dn, getData(), jn(), kn, pn(), Rn, setData()

### Community 25 - "jupyterlab-manager/package.json"
Cohesion: 0.04
Nodes (51): author, bugs, url, description, files, gitHead, homepage, rimraf (+43 more)

### Community 27 - "7032"
Cohesion: 0.09
Nodes (39): 7032(), a(), b(), e1(), e2(), ea(), ec(), ed() (+31 more)

### Community 28 - "get_joint_positions"
Cohesion: 0.08
Nodes (48): all_between(), all_close(), compute_jacobian(), compute_joint_weights(), control_joints(), control_joints_hold(), count_until(), create_sub_robot() (+40 more)

### Community 29 - "MIDAS.py"
Cohesion: 0.07
Nodes (27): BaseModel, ConvTransposeNorm, DPT, DPTDepthModel, FeatureFusionBlock_custom, forward_levit(), get_activation(), Interpolate (+19 more)

### Community 30 - "543.1c80835729e2b26eb76d.js"
Cohesion: 0.06
Nodes (27): Et(), _f(), Ge(), he(), ie(), Jr(), Mf(), Mt() (+19 more)

### Community 31 - "index.js"
Cohesion: 0.06
Nodes (23): Cn(), Bt(), ci(), co(), dn(), dr(), Ft(), jo() (+15 more)

### Community 32 - "add_line"
Cohesion: 0.05
Nodes (41): add_line(), add_text(), connect(), disable_preview(), disconnect(), draw_aabb(), draw_collision_info(), draw_link_name() (+33 more)

### Community 33 - "me"
Cohesion: 0.05
Nodes (6): fe(), v(), me, qe(), rt(), w()

### Community 34 - "multiply"
Cohesion: 0.07
Nodes (41): add_body_name(), add_fixed_constraint(), add_pose_constraint(), approach_from_grasp(), Attachment, body_from_end_effector(), create_attachment(), create_flying_body() (+33 more)

### Community 35 - "open3d/package.json"
Cohesion: 0.04
Nodes (44): author, extension, load, dependencies, @jupyter-widgets/base, lodash, webrtc-adapter, description (+36 more)

### Community 36 - "upsamplers.py"
Cohesion: 0.07
Nodes (12): AdaptiveConv, Bilinear, CarafeUpsampler, get_upsampler(), IFA, JBULearnedRange, JBUStack, LayeredResizeConv (+4 more)

### Community 37 - "dump_body"
Cohesion: 0.07
Nodes (43): body_from_name(), clone_world(), dump_body(), dump_joint(), dump_link(), dump_world(), get_adjacent_fixed_links(), get_bodies() (+35 more)

### Community 40 - "plan_nonholonomic_motion"
Cohesion: 0.07
Nodes (39): adjust_path(), check_initial_end(), dict_from_kwargs(), discretize_curve(), get_acceleration_fn(), get_base_difference_fn(), get_base_distance_fn(), get_closest_angle_fn() (+31 more)

### Community 41 - "ee"
Cohesion: 0.07
Nodes (3): ee, lt(), Rt()

### Community 43 - "get_all_links"
Cohesion: 0.07
Nodes (38): aabb_overlap(), aabb_union(), any_link_pair_collision(), are_links_adjacent(), body_collision(), cached_fn(), can_collide(), child_link_from_joint() (+30 more)

### Community 45 - "x"
Cohesion: 0.06
Nodes (5): hide(), pt, show(), x(), yt()

### Community 46 - "train_jbu_upsampler.py"
Cohesion: 0.09
Nodes (14): apply_jitter(), JitteredImage, Dataset, sample_transform(), AttentionDownsampler, SimpleDownsampler, entropy(), JBUFeatUp (+6 more)

### Community 48 - "Saver"
Cohesion: 0.06
Nodes (12): BodySaver, ClientSaver, ConfSaver, get_velocity(), PoseSaver, Profiler, Saver, set_client() (+4 more)

### Community 49 - ".addClass"
Cohesion: 0.09
Nodes (3): H(), Nt(), Ot()

### Community 52 - "BaseTrainTester"
Cohesion: 0.08
Nodes (15): BaseTrainTester, no_grad, Run a single training step., Run a given number of evaluation steps., Load from checkpoint., Save checkpoint if requested., Train/test a trajectory optimization algorithm., Basic train/test class to be inherited. (+7 more)

### Community 53 - "models.py"
Cohesion: 0.10
Nodes (22): KeypointDataset, Dataset, Adapted from https://github.com/nickgkan/3d_diffuser_actor Args: traj: T 8 (3…, Interpolate a trajectory to have a fixed length., Args: traj_qpos: valid_delta: Returns: Array of valid entry ids, Args: traj: (T, 4, 4), TrajectoryInterpolator, visualize_trajectory() (+14 more)

### Community 54 - "clone_body"
Cohesion: 0.11
Nodes (35): aabb_from_oobb(), aabb_from_points(), clone_body(), clone_collision_shape(), clone_visual_shape(), collision_shape_from_data(), get_aabb_vertices(), get_client() (+27 more)

### Community 55 - "84.09c61a35dce1705f.js"
Cohesion: 0.08
Nodes (18): 1051(), 2780(), 3231(), 3531(), 3698(), 6069(), 609(), 611() (+10 more)

### Community 56 - "d"
Cohesion: 0.11
Nodes (34): f(), t(), x(), S(), iC(), ij(), iv(), iw() (+26 more)

### Community 57 - "block.py"
Cohesion: 0.11
Nodes (19): Attention, MemEffAttention, Tensor, add_residual(), Block, drop_add_residual_stochastic_depth(), drop_add_residual_stochastic_depth_list(), get_attn_bias_and_cat() (+11 more)

### Community 58 - "PybulletRobot"
Cohesion: 0.08
Nodes (22): adjacent_from_edges(), control_joint(), custom_limits_from_base_limits(), get_adjacent_links(), get_connected_components(), get_fixed_links(), get_full_configuration(), get_joint() (+14 more)

### Community 59 - ".dispose"
Cohesion: 0.08
Nodes (3): clearData(), unregister_target(), ve

### Community 60 - "featup/util.py"
Cohesion: 0.10
Nodes (20): Returns the tokenized representation of given input string(s) Parameters…, tokenize(), plot_feats(), plot_lang_heatmaps(), no_grad, Tensor, sample(), implicit_feats() (+12 more)

### Community 61 - "losses.py"
Cohesion: 0.09
Nodes (15): compute_scale_and_shift(), gradient_loss(), GradientLoss, mse_loss(), MSELoss, reduction_batch_based(), reduction_image_based(), SampledCRFLoss (+7 more)

### Community 62 - "datasets/util.py"
Cohesion: 0.08
Nodes (14): bit_get(), Coco, create_pascal_label_colormap(), Dataset, Gets the bit value. Args: val: Input value, int or numpy int array. idx: Which…, Creates a label colormap used in PASCAL VOC segmentation benchmark. Returns: A…, DAVIS, Dataset (+6 more)

### Community 63 - "MaskedAutoencoderViT"
Cohesion: 0.09
Nodes (15): get_1d_sincos_pos_embed_from_grid(), get_2d_sincos_pos_embed(), get_2d_sincos_pos_embed_from_grid(), MaskedAutoencoderViT, PatchEmbed, Tensor, 2D Image to Patch Embedding, Masked Autoencoder with VisionTransformer backbone (+7 more)

### Community 64 - "c"
Cohesion: 0.07
Nodes (4): c(), hn(), On, zn

### Community 66 - ".__init__"
Cohesion: 0.10
Nodes (11): Conv1dBlock, Downsample1d, KpEncoder, LinearAttention, PreNorm, x : [ batch_size x inp_channels x horizon ] t : [ batch_size x embed_dim ]…, Conv1d --> GroupNorm --> Mish, Residual (+3 more)

### Community 67 - "get_length"
Cohesion: 0.08
Nodes (31): angle_between(), clip(), clip_pixel(), convex_area(), convex_centroid(), convex_combination(), convex_signed_area(), get_closest_edge_point() (+23 more)

### Community 68 - "get_image"
Cohesion: 0.07
Nodes (24): compiled_with_numpy(), compute_camera_pose(), compute_view_matrix(), demask_pixel(), extract_segmented(), get_angle(), get_camera(), get_camera_matrix() (+16 more)

### Community 69 - "get_featurizer"
Cohesion: 0.08
Nodes (8): CLIPFeaturizer, DeepLabV3Featurizer, DINOv2Featurizer, MAEFeaturizer, MaskCLIPFeaturizer, MIDASFeaturizer, ResNetFeaturizer, get_featurizer()

### Community 70 - "gu"
Cohesion: 0.10
Nodes (28): ae(), ai(), Bt(), Bu(), di(), _e(), ei(), fr() (+20 more)

### Community 72 - "DINO.py"
Cohesion: 0.11
Nodes (12): Attention, Block, DINOFeaturizer, drop_path(), DropPath, Mlp, PatchEmbed, Image to Patch Embedding (+4 more)

### Community 73 - "dependencies"
Cohesion: 0.07
Nodes (27): dependencies, jquery, @jupyter-widgets/base, @jupyter-widgets/base-manager, @jupyter-widgets/controls, @jupyter-widgets/output, @jupyterlab/application, @jupyterlab/apputils (+19 more)

### Community 74 - "o"
Cohesion: 0.09
Nodes (3): ce(), o(), spacesAndCommentsFromStart()

### Community 75 - "2888"
Cohesion: 0.12
Nodes (16): 1374(), T(), 2888(), S(), t(), w(), 3379(), v() (+8 more)

### Community 76 - "clip.py"
Cohesion: 0.12
Nodes (16): device, available_models(), _convert_image_to_rgb(), _download(), load(), Load a CLIP model Parameters ---------- name : str A model name listed by…, Returns the names of available CLIP models, _transform() (+8 more)

### Community 77 - "160.235911d8e26dc735.js"
Cohesion: 0.12
Nodes (13): 1233(), 2591(), i(), r(), 4400(), 4651(), ee(), ei() (+5 more)

### Community 78 - "train_implicit_upsampler.py"
Cohesion: 0.13
Nodes (10): ImplicitFeaturizer, MinMaxScaler, ExplicitUpsampler, get_implicit_upsampler(), mag(), my_app(), DictConfig, main (+2 more)

### Community 79 - "DINOv2.py"
Cohesion: 0.14
Nodes (9): BlockChunk, DinoVisionTransformer, Tensor, Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64, Args: img_size (int, tuple): input image size patch_size (int, tuple): patch…, vit_base(), vit_giant2(), vit_large() (+1 more)

### Community 80 - "VGG"
Cohesion: 0.15
Nodes (17): make_layers(), VGG 11-layer model (configuration "A") Args: pretrained (bool): If True,…, VGG 11-layer model (configuration "A") with batch normalization Args:…, VGG 13-layer model (configuration "B") Args: pretrained (bool): If True,…, VGG 13-layer model (configuration "B") with batch normalization Args:…, VGG 16-layer model (configuration "D") with batch normalization Args:…, VGG 19-layer model (configuration "E") Args: pretrained (bool): If True,…, VGG 19-layer model (configuration 'E') with batch normalization Args:… (+9 more)

### Community 81 - "n"
Cohesion: 0.14
Nodes (23): ci(), be(), Dt(), fa(), fi(), gi(), go(), hf() (+15 more)

### Community 82 - "layers/__init__.py"
Cohesion: 0.12
Nodes (10): _build_mlp(), DINOHead, Mlp, Module, Tensor, Module, Tensor, SwiGLUFFN (+2 more)

### Community 83 - ".__init__"
Cohesion: 0.17
Nodes (10): AdaptiveAvgPool2d, BatchNorm2d, Conv2d, forward_hook(), Linear, ReLU, Sequential, conv1x1() (+2 more)

### Community 84 - "ResNet"
Cohesion: 0.12
Nodes (13): BasicBlock, Bottleneck, Constructs a ResNet-18 model. Args: pretrained (bool): If True, returns a model…, Constructs a ResNet-34 model. Args: pretrained (bool): If True, returns a model…, Constructs a ResNet-50 model. Args: pretrained (bool): If True, returns a model…, Constructs a ResNet-101 model. Args: pretrained (bool): If True, returns a…, Constructs a ResNet-152 model. Args: pretrained (bool): If True, returns a…, ResNet (+5 more)

### Community 85 - "create_shape"
Cohesion: 0.11
Nodes (22): apply_alpha(), assign_link_colors(), create_body(), create_box(), create_capsule(), create_collision_shape(), create_cylinder(), create_faces() (+14 more)

### Community 86 - "i"
Cohesion: 0.15
Nodes (22): as(), cr(), Ea(), eo(), Gi(), i(), ia(), io() (+14 more)

### Community 87 - "main_kalm_eval_robot.py"
Cohesion: 0.12
Nodes (16): dict, kalm_drawer_config, object, ModelInferenceWrapper, no_grad, Args: kp_detection_ret Returns: predicted_traj: T 4 4 (model frame)…, initialize_robot_policy(), get_pcd() (+8 more)

### Community 88 - "jupyterlab-plotly/package.json"
Cohesion: 0.10
Nodes (20): author, dependencies, lodash-es, @lumino/widgets, plotly.js, description, @jupyter/builder, @lumino/widgets (+12 more)

### Community 89 - "KeypointPredictor"
Cohesion: 0.20
Nodes (11): ImageRecord, KeypointPredictor, MatchingAlgoParams, ndarray, no_grad, object, tensor, Args: image1: image2: image2_invalidmask: dino_weight: vis: guided_mask: Guided… (+3 more)

### Community 90 - "n"
Cohesion: 0.13
Nodes (13): 1138(), 2743(), a(), i(), n(), 2809(), 4291(), 5930() (+5 more)

### Community 92 - "KalmDiffuser"
Cohesion: 0.18
Nodes (8): KalmDiffuser, KpEncodedInput, tensor, Arguments: keypoints_feat: (B, n_kp, feat_dim) keypoints_xyz: (B, n_kp, 3)…, DiffusionHeadOutDict, object, x : [ batch x horizon x transition ], TemporalUnet_diffuser

### Community 93 - "a"
Cohesion: 0.14
Nodes (12): 1206(), c(), u(), 2315(), 2357(), h(), 7991(), l() (+4 more)

### Community 94 - "de"
Cohesion: 0.16
Nodes (18): At(), Cf(), de(), ji(), ki(), Le(), Lt(), mu() (+10 more)

### Community 95 - "Qa"
Cohesion: 0.18
Nodes (17): a, ar(), bo(), go(), Hn(), ho(), ir(), Is() (+9 more)

### Community 96 - "ss"
Cohesion: 0.14
Nodes (18): ao(), Bs(), cn(), di(), Fs(), gs(), $n(), ni() (+10 more)

### Community 97 - "HighResEmbs.py"
Cohesion: 0.17
Nodes (11): clamp_mag(), generate_subset(), HighResEmb, HighResEmbHelper, load_hr_emb(), load_some_hr_feats(), mag(), model_collate() (+3 more)

### Community 98 - "modules/layers.py"
Cohesion: 0.18
Nodes (10): Clone, ConvTranspose2d, Dropout, einsum, GELU, Identity, LayerNorm, LeakyReLU (+2 more)

### Community 99 - "s"
Cohesion: 0.15
Nodes (17): 1182(), 1340(), c(), 3212(), s(), 353(), 358(), ey() (+9 more)

### Community 100 - "devDependencies"
Cohesion: 0.12
Nodes (16): devDependencies, eslint, eslint-config-prettier, eslint-plugin-prettier, @jupyter/builder, @jupyterlab/cells, npm-run-all, prettier (+8 more)

### Community 102 - "m"
Cohesion: 0.17
Nodes (9): 5193(), d(), 5482(), p(), 7521(), d(), h(), m() (+1 more)

### Community 103 - "div_round_up"
Cohesion: 0.22
Nodes (13): adaptive_conv_forward(), adaptive_conv_grad_filters(), adaptive_conv_grad_input(), m, Tensor, TORCH_EXTENSION_NAME, PYBIND11_MODULE(), adaptive_conv_cuda_forward() (+5 more)

### Community 104 - "EmbeddingFile"
Cohesion: 0.15
Nodes (4): EmbeddingAndImage, EmbeddingFile, Dataset, modified from:…

### Community 105 - "ChannelNorm"
Cohesion: 0.22
Nodes (11): ChannelNorm, clip(), dino16(), dinov2(), _load_backbone(), maskclip(), Module, The function that will be called by Torch Hub users to instantiate your model.… (+3 more)

### Community 106 - "scripts"
Cohesion: 0.13
Nodes (15): scripts, build, build:labextension, build:labextension:dev, build:lib, build:prod, clean, clean:all (+7 more)

### Community 107 - "t"
Cohesion: 0.17
Nodes (11): 1467(), n(), e(), 8052(), a(), o(), 8287(), t() (+3 more)

### Community 108 - "9907"
Cohesion: 0.23
Nodes (15): 1280(), 2841(), 4004(), 583(), 7476(), 8668(), 8854(), 9907() (+7 more)

### Community 110 - "RelPropSimple"
Cohesion: 0.26
Nodes (6): Add, AvgPool2d, Multiply, RelPropSimple, safe_divide(), ZeroPad2d

### Community 111 - "mask_proposal.py"
Cohesion: 0.26
Nodes (9): clean_mask(), MaskProposal, normalize_label(), Any, ndarray, Remove tiny components and small mask defects., save_masks(), visualize_masks() (+1 more)

### Community 112 - "fu"
Cohesion: 0.22
Nodes (14): ar(), Bi(), ce(), cu(), df(), du(), fu(), hu() (+6 more)

### Community 113 - "je"
Cohesion: 0.25
Nodes (14): Do(), ee(), ef(), ff(), Ho(), iu(), je(), nf() (+6 more)

### Community 114 - "e"
Cohesion: 0.19
Nodes (14): ac(), Dt(), e(), jt(), li(), Lt(), Mr(), nr() (+6 more)

### Community 115 - "ai"
Cohesion: 0.22
Nodes (12): ai(), bi(), ds(), ei(), fi(), fn(), ji(), ln() (+4 more)

### Community 116 - "get_region_proposals"
Cohesion: 0.27
Nodes (11): clean_json_response(), encode_image(), get_region_proposals(), Any, Convert an image file into a data URL for the vision API., Remove accidental markdown fences from the model response., Clamp and validate model-produced bounding boxes., Save reasoning/region output as JSON. (+3 more)

### Community 117 - "3018"
Cohesion: 0.21
Nodes (7): 3018(), a(), d(), g(), m(), p(), v()

### Community 118 - "1944"
Cohesion: 0.17
Nodes (3): 1944(), m(), p()

### Community 120 - "trunc_normal_"
Cohesion: 0.20
Nodes (7): DINOHead, _no_grad_trunc_normal_(), trunc_normal_(), init_weights_vit_timm(), named_apply(), Module, ViT weight initialization, original timm impl (for reproducibility)

### Community 121 - "s"
Cohesion: 0.18
Nodes (10): 199(), 3300(), o(), 5517(), 6825(), l(), 833(), d() (+2 more)

### Community 124 - "Ct"
Cohesion: 0.26
Nodes (12): cr(), Ct(), eo(), b(), d(), g(), y(), ir() (+4 more)

### Community 126 - "gf"
Cohesion: 0.18
Nodes (7): gf(), hr(), Ht(), pf(), pu(), a(), bn()

### Community 127 - "es"
Cohesion: 0.22
Nodes (11): Bn(), es(), hi(), Ka(), ns(), on(), rs(), un() (+3 more)

### Community 128 - "extract_tag_content"
Cohesion: 0.24
Nodes (5): extract_tag_content(), Extract the content of a tag from a string. Args: content: str. The content…, resize_to(), ndarray, Prompt GPT to obtain an x,y coordinate of the task relevant part. Args:…

### Community 129 - "VGG_spread"
Cohesion: 0.20
Nodes (5): MaxPool2d, make_layers_list(), VGG 16-layer model (configuration "D") Args: pretrained (bool): If True,…, vgg16_spread(), VGG_spread

### Community 132 - "mi"
Cohesion: 0.27
Nodes (10): br(), Fr(), Gr(), hr(), mi(), mo(), nc(), po() (+2 more)

### Community 133 - "oo"
Cohesion: 0.24
Nodes (10): Fo(), ga(), oo(), Pa(), ps(), Ut(), wa(), wo() (+2 more)

### Community 134 - "or"
Cohesion: 0.33
Nodes (3): hs(), or(), s

### Community 135 - "SingleImageDataset"
Cohesion: 0.25
Nodes (3): Dataset, SingleImageDataset, SlicedDataset

### Community 136 - "af"
Cohesion: 0.31
Nodes (9): af(), ea(), eu(), fe(), Jt(), ju(), pe(), ru() (+1 more)

### Community 137 - "za"
Cohesion: 0.25
Nodes (9): er(), Ht(), ii(), ki(), pi(), qn(), rr(), za() (+1 more)

### Community 138 - "adaptive_conv.cpp"
Cohesion: 0.32
Nodes (7): adaptive_conv_forward(), adaptive_conv_grad_filters(), adaptive_conv_grad_input(), m, Tensor, TORCH_EXTENSION_NAME, PYBIND11_MODULE()

### Community 140 - "PatchEmbed"
Cohesion: 0.29
Nodes (4): make_2tuple(), PatchEmbed, Tensor, 2D image to patch embedding: (B,C,H,W) -> (B,N,D) Args: img_size: Image size.…

### Community 141 - "tc"
Cohesion: 0.29
Nodes (8): E(), ee(), eo(), rg(), rv(), t9(), tc(), tk()

### Community 142 - "6439"
Cohesion: 0.29
Nodes (5): 2305(), 612(), u(), 6439(), h()

### Community 143 - "_i"
Cohesion: 0.25
Nodes (8): au(), Gr(), Gt(), hi(), _i(), If(), qe(), Qo()

### Community 144 - "jupyterlab"
Cohesion: 0.29
Nodes (7): extension, load, jupyterlab, _build, extension, outputDir, schemaDir

### Community 145 - "jupyterlab"
Cohesion: 0.29
Nodes (7): load, mimeExtension, jupyterlab, _build, mimeExtension, outputDir, webpackConfig

### Community 146 - "scripts"
Cohesion: 0.29
Nodes (7): scripts, build, build:labextension, build:mimerenderer, build:widget, typecheck, watch

### Community 148 - "devDependencies"
Cohesion: 0.40
Nodes (5): devDependencies, esbuild, @jupyter/builder, @types/plotly.js, typescript

### Community 149 - "0.e889754e73c03c32.js"
Cohesion: 0.40
Nodes (4): TODO: possible optimizations: avoid overcalculating all for vertices and calc…, TODO: precalculate dot products, normalize things beforehead etc., TODO: refactor to rectangular algorithm, FIXME: there should be more elegant solution

### Community 150 - "A"
Cohesion: 0.50
Nodes (4): A(), eq(), ra(), rc()

### Community 154 - "repository"
Cohesion: 0.67
Nodes (3): repository, type, url

## Knowledge Gaps
- **180 isolated node(s):** `KeypointDistillArguments`, `TrainerArguments`, `eslint`, `eslint-config-prettier`, `eslint-plugin-prettier` (+175 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1735 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `2()` connect `2` to `i`, `a`, `n`, `o`, `368`, `0.e889754e73c03c32.js`, `d`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `s()` connect `i` to `2`, `a`, `d`, `o`, `d`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `_` connect `_` to `ne`, `qt`, `fe`, `d`, `g`, `k`, `n`, `.update`, `ge`, `.touch`, `re`, `.defaults`, `l`, `kn`, `te`, `index.js`, `me`, `Ae`, `T`, `ee`, `y`, `x`, `a`, `.addClass`, `i`, `b`, `.dispose`, `c`, `m`, `et`, `o`, `be`, `we`, `dt`, `it`, `gf`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `2()` (e.g. with `eC()` and `e0()`) actually correct?**
  _`2()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 57 inferred relationships involving `o()` (e.g. with `a()` and `c()`) actually correct?**
  _`o()` has 57 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `i()` (e.g. with `a()` and `n()`) actually correct?**
  _`i()` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `a()` (e.g. with `a()` and `b()`) actually correct?**
  _`a()` has 65 INFERRED edges - model-reasoned connections that need verification._
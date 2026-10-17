"""The paper's configuration, frozen: the corpora and their expert trios, the protocol, the hyperparameter spaces
(Table 2), and the configuration each search selected.

`MAIN[corpus]` is everything a main notebook needs; `SPACES[arm]` is the space every corpus searches; and
`SELECTED[corpus][arm]` is the configuration the reported search kept (`SELECTED_CE`: the cross-entropy controls of
the loss ablation, Table 5b). A search keeps its lowest hold-out WER, and when
several trials tie on it, the reported run kept one of them: `SELECTED` names which, and `fastt.experiment` checks
that it is one of the best trials of the stored search before using it.
"""

__all__ = ['MainConfig', 'MAIN', 'SPACES', 'SELECTED', 'SELECTED_CE', 'SELECTION_ARMS', 'BASELINE_ARMS', 'ARMS',
           'TITLES']

from dataclasses import dataclass, field

from fastt.tuning import Param

SELECTION_ARMS = ('fastt_alternating', 'fastt_boosted', 'adastt', 'raw_sdt', 'kbest_xgboost', 'kbest_sdt')
BASELINE_ARMS = ('rover_w', 'rover_conf', 'cn_mbr', 'score_fusion', 'argmin_clf', 'probe', 'soft_probe', 'qe_router')
ARMS = SELECTION_ARMS + BASELINE_ARMS


@dataclass(frozen=True)
class MainConfig:
    """One corpus's main experiment. The seeds are separate fields on purpose: the hold-out cut, the search sampler,
    the fold structure and the estimator initialisation are different randomisations."""

    dataset: str
    title: str
    members: tuple                                   # the three experts
    pooling: str = 'mean'
    seconds_per_frame: float = 0.02     # encoder stride; only for the hours estimate
    holdout_frac: float = 0.25           # share of the clips the search may see
    inner_val_frac: float = 0.3          # share of the hold-out a configuration is scored on
    achievable_margin: float = 2.0
    search_budget: int = 10              # random draws per arm
    sampler: str = 'random'
    search_rounds: int = 0
    search_name_suffix: str = ''
    arm_versions: dict = field(default_factory=dict)
    partition_seed: int = 20260901         # the hold-out / evaluation cut
    study_seed: int = 7                  # the random-search sampler
    model_seed: int = 42                 # every estimator fit
    cv_random_seed: int = 20260821         # the Random-Ensemble arm
    cv_seeds: tuple = (11, 22, 33, 44, 55)   # the five 5x2 repetitions
    nthread: int = 4                    # XGBoost reproduces only at a fixed thread count
    alpha: float = 0.05


TITLES = {'librispeech': 'LibriSpeech', 'voxpopuli': 'VoxPopuli', 'ami_sdm': 'AMI-SDM', 'earnings22': 'Earnings-22', 'peoples_speech': "People's Speech"}

MAIN = {
    "librispeech": MainConfig("librispeech", "LibriSpeech", ('parakeet_tdt_0_6b_v2', 'canary_qwen_2_5b', 'whisper_large_v3')),
    "voxpopuli": MainConfig("voxpopuli", "VoxPopuli", ('espnet_transducer', 'espnet_transformer', 'data2vec')),
    "ami_sdm": MainConfig("ami_sdm", "AMI-SDM", ('espnet_transducer', 'espnet_transformer', 'wav2vec2_large')),
    "earnings22": MainConfig("earnings22", "Earnings-22", ('data2vec', 'espnet_transducer', 'wav2vec2_large')),
    "peoples_speech": MainConfig("peoples_speech", "People's Speech", ('data2vec', 'espnet_transformer', 'wav2vec2')),
}

SPACES = {}

SPACES["fastt_alternating"] = {p.name: p for p in [
    Param('transform_type', 'cat', choices=('nonlinear', 'linear', 'diagonal', 'low_rank', 'identity')),
    Param('num_iterations', 'cat', choices=(6, 12, 3, 1)),
    Param('transform_lr', 'cat', choices=(0.001, 0.0001, 0.01)),
    Param('transform_steps', 'cat', choices=(100, 25, 400)),
    Param('surrogate_lr', 'cat', choices=(0.01, 0.1, 0.001)),
    Param('surrogate_steps', 'cat', choices=(50, 200, 400)),
    Param('lambda1', 'cat', choices=(0.1, 0.001, 1e-05)),
    Param('lambda2', 'cat', choices=(1e-05, 0.1, 0.001)),
    Param('sel_n_estimators', 'cat', choices=(400,)),
    Param('sel_learning_rate', 'cat', choices=(0.03, 0.3, 0.1, 0.01)),
    Param('sel_max_depth', 'cat', choices=(5, 12, 8, 3)),
    Param('sel_subsample', 'cat', choices=(1.0, 0.6, 0.8)),
    Param('sel_colsample', 'cat', choices=(0.6, 1.0, 0.3, 0.1)),
    Param('sel_min_child_weight', 'cat', choices=(1.0,)),
    Param('sel_reg_lambda', 'cat', choices=(3.0,)),
    Param('weight_decay', 'cat', choices=(0.01,)),
    Param('out_ratio', 'cat', choices=(1.0, 0.1, 0.3)),
    Param('bottleneck_ratio', 'cat', choices=(0.05,)),
]}

SPACES["fastt_boosted"] = {p.name: p for p in [
    Param('transform_type', 'cat', choices=('identity', 'low_rank', 'linear', 'nonlinear', 'diagonal')),
    Param('lambda1', 'cat', choices=(0.001,)),
    Param('lambda2', 'cat', choices=(0.001, 1e-05, 0.1)),
    Param('out_ratio', 'cat', choices=(0.1,)),
    Param('bottleneck_ratio', 'cat', choices=(0.15, 0.5, 0.05)),
    Param('num_rounds', 'cat', choices=(6, 1, 4, 2)),
    Param('num_trees', 'cat', choices=(24, 8, 16, 4)),
    Param('depth', 'cat', choices=(7,)),
    Param('lmbda', 'cat', choices=(0.01, 0.0001, 1.0, 1e-06)),
    Param('lmbda2', 'cat', choices=(1e-07,)),
    Param('adam_lr', 'cat', choices=(0.0003,)),
    Param('adam_weight_decay', 'cat', choices=(0.001, 1e-07, 1e-05, 0.1)),
    Param('batch_size', 'cat', choices=(512, 128, 2048)),
    Param('patience', 'cat', choices=(40, 20, 10)),
    Param('t_weight_decay', 'cat', choices=(1e-06,)),
]}

SPACES["adastt"] = {p.name: p for p in [
    Param('sel_n_estimators', 'cat', choices=(1000, 400, 150, 50)),
    Param('sel_learning_rate', 'cat', choices=(0.3, 0.1, 0.03, 0.01)),
    Param('sel_max_depth', 'cat', choices=(3,)),
    Param('sel_subsample', 'cat', choices=(1.0, 0.8, 0.6)),
    Param('sel_colsample', 'cat', choices=(1.0,)),
    Param('sel_min_child_weight', 'cat', choices=(16.0,)),
    Param('sel_reg_lambda', 'cat', choices=(3.0, 0.3, 30.0)),
]}

SPACES["raw_sdt"] = {p.name: p for p in [
    Param('transform_type', 'cat', choices=('identity',), frozen='identity'),
    Param('t_weight_decay', 'cat', choices=(1e-06,)),
    Param('lambda1', 'cat', choices=(0.001,)),
    Param('lambda2', 'cat', choices=(0.001,)),
    Param('out_ratio', 'cat', choices=(0.1,)),
    Param('bottleneck_ratio', 'cat', choices=(0.05,)),
    Param('num_rounds', 'cat', choices=(1, 2, 4, 6)),
    Param('num_trees', 'cat', choices=(8,)),
    Param('depth', 'cat', choices=(2, 7, 5, 3)),
    Param('lmbda', 'cat', choices=(1e-06, 1.0, 0.0001, 0.01)),
    Param('lmbda2', 'cat', choices=(0.1,)),
    Param('adam_lr', 'cat', choices=(0.001, 0.003, 0.0003, 0.01)),
    Param('adam_weight_decay', 'cat', choices=(1e-07,)),
    Param('batch_size', 'cat', choices=(2048, 512, 128)),
    Param('patience', 'cat', choices=(20, 40, 10)),
]}

SPACES["kbest_xgboost"] = {p.name: p for p in [
    Param('sel_n_estimators', 'cat', choices=(150, 1000, 400, 50)),
    Param('sel_learning_rate', 'cat', choices=(0.01,)),
    Param('sel_max_depth', 'cat', choices=(3,)),
    Param('sel_subsample', 'cat', choices=(1.0, 0.6, 0.8)),
    Param('sel_colsample', 'cat', choices=(0.3, 0.1, 0.6, 1.0)),
    Param('sel_min_child_weight', 'cat', choices=(16.0,)),
    Param('sel_reg_lambda', 'cat', choices=(30.0,)),
    Param('k', 'cat', choices=(2048, 512, 128, 32)),
]}

SPACES["kbest_sdt"] = {p.name: p for p in [
    Param('transform_type', 'cat', choices=('identity',), frozen='identity'),
    Param('t_weight_decay', 'cat', choices=(1e-06,)),
    Param('lambda1', 'cat', choices=(0.001,)),
    Param('lambda2', 'cat', choices=(0.001,)),
    Param('out_ratio', 'cat', choices=(0.1,)),
    Param('bottleneck_ratio', 'cat', choices=(0.05,)),
    Param('num_rounds', 'cat', choices=(2, 4, 6, 1)),
    Param('num_trees', 'cat', choices=(16, 8, 24, 4)),
    Param('depth', 'cat', choices=(3, 5, 7, 2)),
    Param('lmbda', 'cat', choices=(1e-06, 0.0001, 1.0, 0.01)),
    Param('lmbda2', 'cat', choices=(0.1,)),
    Param('adam_lr', 'cat', choices=(0.001, 0.01, 0.0003, 0.003)),
    Param('adam_weight_decay', 'cat', choices=(1e-07, 0.1, 1e-05, 0.001)),
    Param('batch_size', 'cat', choices=(512, 2048, 128)),
    Param('patience', 'cat', choices=(10, 20, 40)),
    Param('k', 'cat', choices=(32,)),
]}

SPACES["rover_w"] = {p.name: p for p in [
    Param('alpha', 'cat', choices=(1.0,), frozen=1.0),
    Param('conf_source', 'cat', choices=('cn', 'probe', 'blend'), frozen='cn'),
    Param('weighting', 'cat', choices=('inverse_wer', 'log_odds', 'uniform')),
    Param('weight_by_conf', 'cat', choices=(False, True), frozen=False),
    Param('conf_temp', 'cat', choices=(0.25, 1.0, 4.0), frozen=1.0),
    Param('null_penalty', 'cat', choices=(0.25, 0.5)),
]}

SPACES["rover_conf"] = {p.name: p for p in [
    Param('alpha', 'cat', choices=(0.0, 0.25)),
    Param('conf_source', 'cat', choices=('cn',)),
    Param('weighting', 'cat', choices=('inverse_wer', 'log_odds', 'uniform')),
    Param('weight_by_conf', 'cat', choices=(False, True)),
    Param('conf_temp', 'cat', choices=(0.25,)),
    Param('null_penalty', 'cat', choices=(0.25,)),
]}

SPACES["cn_mbr"] = {p.name: p for p in [
    Param('decode', 'cat', choices=('hyp',)),
    Param('candidates', 'cat', choices=('members',)),
    Param('prior', 'cat', choices=('inverse_wer',)),
    Param('temperature', 'cat', choices=(10.0,)),
    Param('null_prior', 'cat', choices=(0.25,)),
    Param('use_conf', 'cat', choices=(False, True)),
    Param('conf_source', 'cat', choices=('blend', 'cn', 'probe')),
]}

SPACES["score_fusion"] = {p.name: p for p in [
    Param('mode', 'cat', choices=('hard',)),
    Param('w_conf', 'cat', choices=(1.0,)),
    Param('w_consensus', 'cat', choices=(-1.0,)),
    Param('w_length', 'cat', choices=(-1.0,)),
    Param('temperature', 'cat', choices=(10.0,)),
    Param('null_penalty', 'cat', choices=(0.25, 1.0, 4.0)),
]}

SPACES["argmin_clf"] = {p.name: p for p in [
    Param('alpha', 'cat', choices=(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)),
]}

SPACES["probe"] = {p.name: p for p in [
    Param('alpha', 'cat', choices=(100.0, 1000.0)),
    Param('objective_weight', 'cat', choices=('length', 'uniform')),
]}

SPACES["soft_probe"] = {p.name: p for p in [
    Param('alpha', 'cat', choices=(10.0, 100.0)),
    Param('objective_weight', 'cat', choices=('length', 'uniform')),
    Param('beta', 'cat', choices=(0.5, 0.1)),
]}

SPACES["qe_router"] = {p.name: p for p in [
    Param('target', 'cat', choices=('errors', 'wer')),
    Param('alpha_lo', 'cat', choices=(0.0001,)),
    Param('alpha_decades', 'cat', choices=(4,)),
    Param('n_alphas', 'cat', choices=(8, 16, 40)),
    Param('standardize', 'cat', choices=(False, True)),
]}

SELECTED = {
    "librispeech": {
        "fastt_alternating": {'transform_type': 'nonlinear', 'num_iterations': 1, 'transform_lr': 0.0001, 'transform_steps': 400, 'surrogate_lr': 0.001, 'surrogate_steps': 50, 'weight_decay': 0.01, 'out_ratio': 0.1, 'bottleneck_ratio': 0.05, 'sel_n_estimators': 400, 'sel_learning_rate': 0.01, 'sel_max_depth': 3, 'sel_subsample': 0.8, 'sel_colsample': 1.0, 'sel_min_child_weight': 1.0, 'sel_reg_lambda': 3.0},
        "fastt_boosted": {'transform_type': 'low_rank', 'num_rounds': 6, 'num_trees': 24, 'depth': 7, 'lmbda': 1e-06, 'lmbda2': 1e-07, 'adam_lr': 0.0003, 'adam_weight_decay': 1e-07, 'batch_size': 512, 'patience': 20, 't_weight_decay': 1e-06, 'out_ratio': 0.1, 'bottleneck_ratio': 0.15, 'epochs': 200},
        "adastt": {'sel_n_estimators': 400, 'sel_learning_rate': 0.03, 'sel_max_depth': 3, 'sel_subsample': 0.6, 'sel_colsample': 1.0, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 0.3},
        "raw_sdt": {'transform_type': 'identity', 'num_rounds': 2, 'num_trees': 8, 'depth': 5, 'lmbda': 0.0001, 'lmbda2': 0.1, 'adam_lr': 0.001, 'adam_weight_decay': 1e-07, 'batch_size': 512, 'patience': 20, 'epochs': 200},
        "kbest_xgboost": {'sel_n_estimators': 1000, 'sel_learning_rate': 0.01, 'sel_max_depth': 3, 'sel_subsample': 0.8, 'sel_colsample': 1.0, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 30.0, 'k': 512},
        "kbest_sdt": {'transform_type': 'identity', 'num_rounds': 4, 'num_trees': 24, 'depth': 7, 'lmbda': 1e-06, 'lmbda2': 0.1, 'adam_lr': 0.003, 'adam_weight_decay': 0.1, 'batch_size': 2048, 'patience': 20, 'epochs': 200, 'k': 32},
        "rover_w": {'alpha': 1.0, 'conf_source': 'cn', 'weighting': 'log_odds', 'weight_by_conf': False, 'null_penalty': 0.5},
        "rover_conf": {'alpha': 0.25, 'conf_source': 'cn', 'weighting': 'inverse_wer', 'weight_by_conf': True, 'null_penalty': 0.25, 'conf_temp': 0.25},
        "cn_mbr": {'decode': 'hyp', 'prior': 'inverse_wer', 'temperature': 10.0, 'null_prior': 0.25, 'use_conf': True, 'candidates': 'members', 'conf_source': 'blend'},
        "score_fusion": {'mode': 'hard', 'w_conf': 1.0, 'w_consensus': -1.0, 'w_length': -1.0},
        "argmin_clf": {'alpha': 1.0},
        "probe": {'alpha': 1000.0, 'objective_weight': 'length'},
        "soft_probe": {'alpha': 10.0, 'objective_weight': 'uniform', 'beta': 0.5},
        "qe_router": {'target': 'errors', 'alpha_lo': 0.0001, 'alpha_decades': 4, 'n_alphas': 40, 'standardize': False},
    },
    "voxpopuli": {
        "fastt_alternating": {'transform_type': 'nonlinear', 'num_iterations': 3, 'transform_lr': 0.0001, 'transform_steps': 400, 'surrogate_lr': 0.1, 'surrogate_steps': 200, 'weight_decay': 0.01, 'out_ratio': 1.0, 'bottleneck_ratio': 0.05, 'sel_n_estimators': 400, 'sel_learning_rate': 0.01, 'sel_max_depth': 3, 'sel_subsample': 0.8, 'sel_colsample': 0.3, 'sel_min_child_weight': 1.0, 'sel_reg_lambda': 3.0},
        "fastt_boosted": {'transform_type': 'low_rank', 'num_rounds': 2, 'num_trees': 16, 'depth': 7, 'lmbda': 1e-06, 'lmbda2': 1e-07, 'adam_lr': 0.0003, 'adam_weight_decay': 1e-07, 'batch_size': 512, 'patience': 20, 't_weight_decay': 1e-06, 'out_ratio': 0.1, 'bottleneck_ratio': 0.15, 'epochs': 200},
        "adastt": {'sel_n_estimators': 400, 'sel_learning_rate': 0.03, 'sel_max_depth': 3, 'sel_subsample': 0.6, 'sel_colsample': 1.0, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 0.3},
        "raw_sdt": {'transform_type': 'identity', 'num_rounds': 2, 'num_trees': 8, 'depth': 5, 'lmbda': 0.0001, 'lmbda2': 0.1, 'adam_lr': 0.001, 'adam_weight_decay': 1e-07, 'batch_size': 512, 'patience': 20, 'epochs': 200},
        "kbest_xgboost": {'sel_n_estimators': 1000, 'sel_learning_rate': 0.01, 'sel_max_depth': 3, 'sel_subsample': 0.8, 'sel_colsample': 1.0, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 30.0, 'k': 512},
        "kbest_sdt": {'transform_type': 'identity', 'num_rounds': 1, 'num_trees': 8, 'depth': 5, 'lmbda': 1e-06, 'lmbda2': 0.1, 'adam_lr': 0.0003, 'adam_weight_decay': 1e-07, 'batch_size': 2048, 'patience': 20, 'epochs': 200, 'k': 32},
        "rover_w": {'alpha': 1.0, 'conf_source': 'cn', 'weighting': 'log_odds', 'weight_by_conf': False, 'null_penalty': 0.5},
        "rover_conf": {'alpha': 0.25, 'conf_source': 'cn', 'weighting': 'inverse_wer', 'weight_by_conf': False, 'null_penalty': 0.25, 'conf_temp': 0.25},
        "cn_mbr": {'decode': 'hyp', 'prior': 'inverse_wer', 'temperature': 10.0, 'null_prior': 0.25, 'use_conf': True, 'candidates': 'members', 'conf_source': 'blend'},
        "score_fusion": {'mode': 'hard', 'w_conf': 1.0, 'w_consensus': -1.0, 'w_length': -1.0},
        "argmin_clf": {'alpha': 0.1},
        "probe": {'alpha': 1000.0, 'objective_weight': 'length'},
        "soft_probe": {'alpha': 100.0, 'objective_weight': 'length', 'beta': 0.5},
        "qe_router": {'target': 'errors', 'alpha_lo': 0.0001, 'alpha_decades': 4, 'n_alphas': 40, 'standardize': False},
    },
    "ami_sdm": {
        "fastt_alternating": {'transform_type': 'diagonal', 'num_iterations': 3, 'transform_lr': 0.01, 'transform_steps': 400, 'surrogate_lr': 0.01, 'surrogate_steps': 400, 'lambda1': 0.1, 'lambda2': 0.001, 'sel_n_estimators': 400, 'sel_learning_rate': 0.1, 'sel_max_depth': 5, 'sel_subsample': 0.6, 'sel_colsample': 0.6, 'sel_min_child_weight': 1.0, 'sel_reg_lambda': 3.0},
        "fastt_boosted": {'transform_type': 'identity', 'num_rounds': 2, 'num_trees': 8, 'depth': 7, 'lmbda': 1.0, 'lmbda2': 1e-07, 'adam_lr': 0.0003, 'adam_weight_decay': 0.001, 'batch_size': 512, 'patience': 20, 'epochs': 200},
        "adastt": {'sel_n_estimators': 400, 'sel_learning_rate': 0.1, 'sel_max_depth': 3, 'sel_subsample': 0.6, 'sel_colsample': 1.0, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 0.3},
        "raw_sdt": {'transform_type': 'identity', 'num_rounds': 2, 'num_trees': 8, 'depth': 5, 'lmbda': 0.0001, 'lmbda2': 0.1, 'adam_lr': 0.001, 'adam_weight_decay': 1e-07, 'batch_size': 512, 'patience': 20, 'epochs': 200},
        "kbest_xgboost": {'sel_n_estimators': 1000, 'sel_learning_rate': 0.01, 'sel_max_depth': 3, 'sel_subsample': 0.8, 'sel_colsample': 1.0, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 30.0, 'k': 128},
        "kbest_sdt": {'transform_type': 'identity', 'num_rounds': 4, 'num_trees': 24, 'depth': 7, 'lmbda': 1e-06, 'lmbda2': 0.1, 'adam_lr': 0.003, 'adam_weight_decay': 0.1, 'batch_size': 2048, 'patience': 20, 'epochs': 200, 'k': 32},
        "rover_w": {'alpha': 1.0, 'conf_source': 'cn', 'weighting': 'inverse_wer', 'weight_by_conf': False, 'null_penalty': 0.5},
        "rover_conf": {'alpha': 0.25, 'conf_source': 'cn', 'weighting': 'inverse_wer', 'weight_by_conf': False, 'null_penalty': 0.25, 'conf_temp': 0.25},
        "cn_mbr": {'decode': 'hyp', 'prior': 'inverse_wer', 'temperature': 10.0, 'null_prior': 0.25, 'use_conf': True, 'candidates': 'members', 'conf_source': 'blend'},
        "score_fusion": {'mode': 'hard', 'w_conf': 1.0, 'w_consensus': -1.0, 'w_length': -1.0},
        "argmin_clf": {'alpha': 0.1},
        "probe": {'alpha': 100.0, 'objective_weight': 'length'},
        "soft_probe": {'alpha': 10.0, 'objective_weight': 'uniform', 'beta': 0.5},
        "qe_router": {'target': 'errors', 'alpha_lo': 0.0001, 'alpha_decades': 4, 'n_alphas': 40, 'standardize': False},
    },
    "earnings22": {
        "fastt_alternating": {'transform_type': 'identity', 'num_iterations': 1, 'sel_n_estimators': 400, 'sel_learning_rate': 0.03, 'sel_max_depth': 5, 'sel_subsample': 0.8, 'sel_colsample': 0.6, 'sel_min_child_weight': 1.0, 'sel_reg_lambda': 3.0},
        "fastt_boosted": {'transform_type': 'diagonal', 'num_rounds': 6, 'num_trees': 24, 'depth': 7, 'lmbda': 0.0001, 'lmbda2': 1e-07, 'adam_lr': 0.0003, 'adam_weight_decay': 1e-07, 'batch_size': 2048, 'patience': 40, 'lambda1': 0.001, 'lambda2': 0.001, 'epochs': 200},
        "adastt": {'sel_n_estimators': 400, 'sel_learning_rate': 0.1, 'sel_max_depth': 3, 'sel_subsample': 1.0, 'sel_colsample': 1.0, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 30.0},
        "raw_sdt": {'transform_type': 'identity', 'num_rounds': 2, 'num_trees': 8, 'depth': 5, 'lmbda': 0.0001, 'lmbda2': 0.1, 'adam_lr': 0.001, 'adam_weight_decay': 1e-07, 'batch_size': 512, 'patience': 20, 'epochs': 200},
        "kbest_xgboost": {'sel_n_estimators': 400, 'sel_learning_rate': 0.01, 'sel_max_depth': 3, 'sel_subsample': 1.0, 'sel_colsample': 0.1, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 30.0, 'k': 512},
        "kbest_sdt": {'transform_type': 'identity', 'num_rounds': 4, 'num_trees': 24, 'depth': 7, 'lmbda': 1e-06, 'lmbda2': 0.1, 'adam_lr': 0.003, 'adam_weight_decay': 0.1, 'batch_size': 2048, 'patience': 20, 'epochs': 200, 'k': 32},
        "rover_w": {'alpha': 1.0, 'conf_source': 'cn', 'weighting': 'log_odds', 'weight_by_conf': False, 'null_penalty': 0.5},
        "rover_conf": {'alpha': 0.25, 'conf_source': 'cn', 'weighting': 'log_odds', 'weight_by_conf': False, 'null_penalty': 0.25, 'conf_temp': 0.25},
        "cn_mbr": {'decode': 'hyp', 'prior': 'inverse_wer', 'temperature': 10.0, 'null_prior': 0.25, 'use_conf': True, 'candidates': 'members', 'conf_source': 'probe'},
        "score_fusion": {'mode': 'hard', 'w_conf': 1.0, 'w_consensus': -1.0, 'w_length': -1.0},
        "argmin_clf": {'alpha': 1.0},
        "probe": {'alpha': 100.0, 'objective_weight': 'length'},
        "soft_probe": {'alpha': 10.0, 'objective_weight': 'uniform', 'beta': 0.5},
        "qe_router": {'target': 'wer', 'alpha_lo': 0.0001, 'alpha_decades': 4, 'n_alphas': 8, 'standardize': True},
    },
    "peoples_speech": {
        "fastt_alternating": {'transform_type': 'linear', 'num_iterations': 12, 'transform_lr': 0.0001, 'transform_steps': 100, 'surrogate_lr': 0.1, 'surrogate_steps': 200, 'weight_decay': 0.01, 'out_ratio': 0.1, 'sel_n_estimators': 400, 'sel_learning_rate': 0.01, 'sel_max_depth': 8, 'sel_subsample': 0.8, 'sel_colsample': 1.0, 'sel_min_child_weight': 1.0, 'sel_reg_lambda': 3.0},
        "fastt_boosted": {'transform_type': 'identity', 'num_rounds': 2, 'num_trees': 8, 'depth': 7, 'lmbda': 1.0, 'lmbda2': 1e-07, 'adam_lr': 0.0003, 'adam_weight_decay': 0.001, 'batch_size': 512, 'patience': 20, 'epochs': 200},
        "adastt": {'sel_n_estimators': 400, 'sel_learning_rate': 0.1, 'sel_max_depth': 3, 'sel_subsample': 1.0, 'sel_colsample': 1.0, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 30.0},
        "raw_sdt": {'transform_type': 'identity', 'num_rounds': 2, 'num_trees': 8, 'depth': 5, 'lmbda': 0.0001, 'lmbda2': 0.1, 'adam_lr': 0.001, 'adam_weight_decay': 1e-07, 'batch_size': 512, 'patience': 20, 'epochs': 200},
        "kbest_xgboost": {'sel_n_estimators': 50, 'sel_learning_rate': 0.01, 'sel_max_depth': 3, 'sel_subsample': 1.0, 'sel_colsample': 0.6, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 30.0, 'k': 32},
        "kbest_sdt": {'transform_type': 'identity', 'num_rounds': 2, 'num_trees': 16, 'depth': 3, 'lmbda': 1.0, 'lmbda2': 0.1, 'adam_lr': 0.001, 'adam_weight_decay': 0.001, 'batch_size': 2048, 'patience': 20, 'epochs': 200, 'k': 32},
        "rover_w": {'alpha': 1.0, 'conf_source': 'cn', 'weighting': 'log_odds', 'weight_by_conf': False, 'null_penalty': 0.5},
        "rover_conf": {'alpha': 0.25, 'conf_source': 'cn', 'weighting': 'log_odds', 'weight_by_conf': False, 'null_penalty': 0.25, 'conf_temp': 0.25},
        "cn_mbr": {'decode': 'hyp', 'prior': 'inverse_wer', 'temperature': 10.0, 'null_prior': 0.25, 'use_conf': True, 'candidates': 'members', 'conf_source': 'probe'},
        "score_fusion": {'mode': 'hard', 'w_conf': 1.0, 'w_consensus': -1.0, 'w_length': -1.0},
        "argmin_clf": {'alpha': 1.0},
        "probe": {'alpha': 1000.0, 'objective_weight': 'length'},
        "soft_probe": {'alpha': 10.0, 'objective_weight': 'uniform', 'beta': 0.5},
        "qe_router": {'target': 'errors', 'alpha_lo': 0.0001, 'alpha_decades': 4, 'n_alphas': 40, 'standardize': False},
    },
}

SELECTED_CE = {
    "ami_sdm": {
        "adastt_ce": {'sel_n_estimators': 400, 'sel_learning_rate': 0.1, 'sel_max_depth': 3, 'sel_subsample': 0.6, 'sel_colsample': 1.0, 'sel_min_child_weight': 16.0, 'sel_reg_lambda': 0.3},
        "fastt_boosted_ce": {'transform_type': 'low_rank', 'num_rounds': 6, 'num_trees': 24, 'depth': 7, 'lmbda': 1e-06, 'lmbda2': 1e-07, 'adam_lr': 0.0003, 'adam_weight_decay': 1e-07, 'batch_size': 512, 'patience': 20, 't_weight_decay': 1e-06, 'out_ratio': 0.1, 'bottleneck_ratio': 0.15, 'epochs': 200},
        "fastt_alternating_ce": {'transform_type': 'low_rank', 'num_iterations': 6, 'transform_lr': 0.001, 'transform_steps': 400, 'surrogate_lr': 0.01, 'surrogate_steps': 200, 'weight_decay': 0.01, 'out_ratio': 0.3, 'bottleneck_ratio': 0.05, 'sel_n_estimators': 400, 'sel_learning_rate': 0.3, 'sel_max_depth': 8, 'sel_subsample': 1.0, 'sel_colsample': 0.1, 'sel_min_child_weight': 1.0, 'sel_reg_lambda': 3.0},
    },
}

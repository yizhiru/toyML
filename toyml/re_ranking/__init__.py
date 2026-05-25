from toyml.re_ranking.dnn.dnn import DNNRankingNetwork
from toyml.re_ranking.dlcm.dlcm import DLCMRankingNetwork

__all__ = [
    'DNNRankingNetwork',
    'DLCMRankingNetwork',
    'PRMRankingNetwork',
]


def __getattr__(name):
    if name == 'PRMRankingNetwork':
        from toyml.re_ranking.prm.prm import PRMRankingNetwork
        return PRMRankingNetwork
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

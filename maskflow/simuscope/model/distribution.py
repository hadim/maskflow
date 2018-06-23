from scipy import stats

from ..serializers import Serializer


class Distribution(Serializer):
    def __init__(self):
        self.distribution_name = None
        self.parameters = {}

    def value(self):
        distrib = getattr(stats, self.distribution_name)
        return distrib.rvs(**self.parameters)

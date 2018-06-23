import logging

from scipy import stats


class AbstractBuilder():

    def __init__(self, model):
        self.model = model
        self._init_builder()

    def _init_builder(self):
        pass

    def build(self, image, verbose=False):
        logging.error("This the builder() function from the AbstractBuilder."
                      "That function should be implemented.")

    def distribution(self, params, size=1):
        distrib = getattr(stats, params["distribution"])
        return distrib.rvs(size=size, **params["parameters"])

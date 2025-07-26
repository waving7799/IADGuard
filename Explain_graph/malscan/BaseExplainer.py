from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    def __init__(self, graphs,X_train_torch, y_train_torch, sen_api_idx, task,label=1):
        self.graphs = graphs
        self.X_train_torch = X_train_torch
        self.y_train_torch = y_train_torch
        self.sen_api_idx = sen_api_idx
        self.type = task
        self.label = label

    @abstractmethod
    def prepare(self, args):
        """Prepars the explanation method for explaining.
        Can for example be used to train the method"""
        pass

    @abstractmethod
    def explain(self, index):
        """
        Main method for explaining samples
        :param index: index of node/graph in self.graphs
        :return: explanation for sample
        """
        pass


import scipy.io
import numpy
import math
import random
from collections.abc import Sequence


class EMDD:

    def __init__(self, training_data):
        self.training_data = training_data

    @staticmethod
    def diverse_density(target, scale, instances):
        x = numpy.tile(numpy.array(target), (len(instances), 1))
        s = numpy.tile(numpy.array(scale), (len(instances), 1))
        B_ij = numpy.array(map(lambda i: i.features, instances))

        D = numpy.mean(numpy.square(s) * numpy.square(B_ij - x), 2)
        P = numpy.exp(numpy.negative(D))

        density = 0
        for i in range(0, len(instances)):
            if instances[i].bag.is_positive():
                if P[i] == 0:
                    P[i] = 1.0e-10

                density -= math.log(P[i])
            else:
                if P[i] == 1:
                    P[i] = 1 - 1.0e-10

                density -= math.log(1 - P[i])

        return density


class MatlabTrainingData:

    def __init__(self, file_name):
        self.training_bags = Bags(scipy.io.loadmat(file_name), "bags", "labels")
        self.test_bags = Bags(scipy.io.loadmat(file_name), "testBags", "testlabels")

    def random_positive_test_bag(self):
        positive_test_bags = [bag for bag in self.test_bags if bag.is_positive()]
        return positive_test_bags[random.randrange(0, len(positive_test_bags))]


class Bags(Sequence):

    def __init__(self, mat, bags_key, labels_key):
        self.bags = []

        mat_bags = mat[bags_key]
        it_bags = numpy.nditer(mat_bags, ["refs_ok", "c_index"])
        while not it_bags.finished:
            mat_bag = mat_bags[0][it_bags.index]
            label = mat[labels_key][0][it_bags.index]

            bag = Bag(it_bags.index, label)
            for instance in mat_bag:
                bag.add_instance(instance)

            self.bags.append(bag)

            it_bags.iternext()

    def __getitem__(self, index):
        return self.bags[index]

    def __len__(self):
        return len(self.bags)


class Bag(Sequence):

    def __init__(self, index, label):
        self.index = index
        self.instances = []
        self.label = label

    def add_instance(self, instance):
        self.instances.append(Instance(instance, self))

    def is_positive(self):
        return self.label == 1

    def random_unused_instance(self):
        unused_instances = [instance for instance in self.instances if instance.used_as_target is False]
        return unused_instances[random.randrange(0, len(unused_instances))]

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)


class Instance:
    """Describes an instance in a bag"""

    def __init__(self, features, bag):
        self.bag = bag
        self.features = features
        self.used_as_target = False


training_data = MatlabTrainingData('training-data/DR_data.mat')


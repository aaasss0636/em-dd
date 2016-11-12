import scipy.io
import scipy.optimize
import numpy
import math
import random
from collections.abc import Sequence
from collections import namedtuple

EMDDResult = namedtuple('EMDDResult', ['target', 'scale', 'density'])


class EMDD:

    def __init__(self, training_data):
        self.training_data = training_data

    def train(self, threshold, perform_scaling=False, runs=1000):
        results = []

        for bag in self.training_data.training_bags:
            for instance in bag.instances:
                instance.used_as_target = False

        for i in range(0, runs):
            print("Run", i)

            random_instance = self.training_data.random_positive_training_bag().random_unused_instance()
            random_instance.used_as_target = True

            results.append(self.run(
                threshold, perform_scaling, random_instance.features, numpy.ones((1, random_instance.features.size))
            ))

        return results

    def run(self, threshold, perform_scaling, target, scale):
        density_difference = math.inf
        previous_density = math.inf
        density = 0

        while density_difference > threshold:

            optimal_instances = []
            for bag in self.training_data.training_bags:
                distances = EMDD.probability_of_closeness(target, scale, bag.instances)
                optimal_instances.append(bag.instances[numpy.argmax(distances)])

            if perform_scaling:
                params = numpy.concatenate((target, scale))
                lower_bound = numpy.zeros(2 * target.size)
                upper_bound = numpy.concatenate((numpy.ones(target.size), numpy.full(target.size, numpy.inf)))

            else:
                params = target
                lower_bound = numpy.zeros(target.size)
                upper_bound = numpy.ones(target.size)

            bounds = tuple(map(lambda x: x, zip(lower_bound, upper_bound)))

            result = scipy.optimize.minimize(
                fun=EMDD.diverse_density,
                x0=params,
                args=optimal_instances,
                bounds=bounds,
                method='L-BFGS-B',
                options={
                    'ftol': 1.0e-03,
                    'maxfun': 50000,
                    'maxiter': 1000,
                }
            )

            params = result.x
            if perform_scaling:
                target = params[1:target.size]
                scale = params[target.size:2 * target.size]
            else:
                target = params
                scale = numpy.ones((1, target.size))

            density = result.fun
            density_difference = previous_density - density
            previous_density = density

            print("Density difference:", density_difference)

        return EMDDResult(target=target, scale=scale, density=density)

    @staticmethod
    def probability_of_closeness(target, scale, instances):
        x = numpy.tile(target, (len(instances), 1))
        s = numpy.tile(scale, (len(instances), 1))
        b_ij = numpy.array(list(map(lambda i: i.features, instances)))

        distances = numpy.mean(numpy.square(s) * numpy.square(b_ij - x), 1)

        return numpy.exp(numpy.negative(distances))

    @staticmethod
    def diverse_density(params, instances):
        if params.size == instances[0].features.size:
            target = params
            scale = numpy.ones((1, instances[0].features.size))
        else:
            target = params[0:instances[0].features.size]
            scale = params[instances[0].features.size:2 * instances[0].features.size]

        p = EMDD.probability_of_closeness(target, scale, instances)

        density = 0
        for i in range(0, len(instances)):
            if instances[i].bag.is_positive():
                if p[i] == 0:
                    p[i] = 1.0e-10

                density -= math.log(p[i])
            else:
                if p[i] == 1:
                    p[i] = 1 - 1.0e-10

                density -= math.log(1 - p[i])

        return density


class MatlabTrainingData:

    def __init__(self, file_name):
        self.training_bags = Bags(scipy.io.loadmat(file_name), "bag", "labels")
        self.test_bags = []
        #self.test_bags = Bags(scipy.io.loadmat(file_name), "testBags", "testlabels")

    def random_positive_training_bag(self):
        positive_training_bags = [bag for bag in self.training_bags if bag.is_positive()]
        return positive_training_bags[random.randrange(0, len(positive_training_bags))]


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


emdd = EMDD(MatlabTrainingData('training-data/synth_data_1.mat'))
results = emdd.train(1.0e-03, perform_scaling=False, runs=100)
print(sorted(results, key=lambda result: result.density)[::-1][0])

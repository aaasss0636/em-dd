import scipy.io
import scipy.optimize
import numpy
import math
import random
import pprint
from collections.abc import Sequence
from collections import namedtuple

EMDDResult = namedtuple('EMDDResult', ['target', 'scale', 'density'])
PredictionResult = namedtuple('PredictionResult', ['bag', 'instances'])

pp = pprint.PrettyPrinter(indent=4)


class EMDD:

    def __init__(self, training_data):
        self.training_data = training_data

    def train(self, threshold, perform_scaling=False, runs=1000):
        results = []

        for bag in self.training_data.training_bags:
            for instance in bag.instances:
                instance.used_as_target = False

        for i in range(0, runs):
            random_instance = self.training_data.random_positive_training_bag().random_unused_instance()
            random_instance.used_as_target = True

            results.append(self.run(
                i, threshold, perform_scaling, random_instance.features, numpy.ones(random_instance.features.size)
            ))

        return results

    def run(self, run, threshold, perform_scaling, target, scale):
        density_difference = math.inf
        previous_density = math.inf
        density = 0

        while density_difference > threshold:

            optimal_instances = []
            for bag in self.training_data.training_bags:
                probabilities = EMDD.positive_instance_probability(target, scale, bag.instances)
                optimal_instances.append(bag.instances[numpy.argmax(probabilities)])

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
                target = params[0:target.size]
                scale = params[target.size:2 * target.size]
            else:
                target = params
                scale = numpy.ones(target.size)

            density = result.fun
            density_difference = abs(previous_density - density)
            previous_density = density

            print("Run ", run, "Target:", target, "Scale:", scale, "Density difference:", density_difference)

        return EMDDResult(target=target, scale=scale, density=density)

    @staticmethod
    def predict(results, bags, threshold, max=True):
        prediction_results = []

        if max is True:
            result = sorted(results, key=lambda r: r.density)[::-1][0]
        else:
            result = sorted(results, key=lambda r: r.density)[0]

        for i in range(0, len(bags)):

            instances = []
            probabilities = EMDD.positive_instance_probability(result.target, result.scale, bags[i].instances)
            for j in range(0, len(probabilities)):
                print("Instance", j, "in bag", i, "has probability", probabilities[j])
                if probabilities[j] > threshold:
                    instances.append(j)

            if len(instances) > 0:
                prediction_results.append(
                    PredictionResult(bag=i, instances=instances)
                )

        return prediction_results

    @staticmethod
    def positive_instance_probability(target, scale, instances):
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

        p = EMDD.positive_instance_probability(target, scale, instances)

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

    def __init__(self, file_name, has_test_data=True):
        self.training_bags = Bags(scipy.io.loadmat(file_name), "bag", "labels")
        if has_test_data is True:
            self.test_bags = Bags(scipy.io.loadmat(file_name), "testBags", "testlabels")

    @staticmethod
    def test_only(file_name):
        return MatlabTrainingData(file_name=file_name, has_test_data=False)

    @staticmethod
    def test_and_training(file_name):
        return MatlabTrainingData(file_name=file_name)

    def random_positive_training_bag(self):
        positive_training_bags = [bag for bag in self.training_bags if bag.is_positive() and bag.is_unused()]
        return positive_training_bags[random.randrange(0, len(positive_training_bags))]


class Bags(Sequence):

    def __init__(self, mat, bags_key, labels_key):
        pp.pprint(mat)

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

    def is_unused(self):
        return len([instance for instance in self.instances if instance.used_as_target is False]) > 0

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


training_data = MatlabTrainingData.test_only('training-data/synth_data_1.mat')
runs = len([bag for bag in training_data.training_bags if bag.is_positive()]) * len(training_data.training_bags[0].instances)

print(runs, "runs")

emdd = EMDD(training_data)
results = emdd.train(1.0e-03, perform_scaling=True, runs=runs)
print(sorted(results, key=lambda result: result.density)[::-1][0])

prediction_results = EMDD.predict(results=results, bags=training_data.training_bags, threshold=0.6, max=True)
print("There are these many results:", len(prediction_results))

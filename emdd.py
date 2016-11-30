import scipy.io
import scipy.optimize
import numpy
import math
import random
import pprint
from collections.abc import Sequence
from collections import namedtuple
from enum import Enum
from functools import reduce

EMDDResult = namedtuple('EMDDResult', ['target', 'scale', 'density'])
PredictionResult = namedtuple('PredictionResult', ['bag', 'instances'])

pp = pprint.PrettyPrinter(indent=4)


class Aggregate(Enum):
    min = 1
    max = 2
    avg = 3


class EMDD:

    def __init__(self, training_data):
        self.training_data = training_data

    @staticmethod
    def predict(results, bags, threshold, aggregate=Aggregate.avg):
        prediction_results = []

        if aggregate == Aggregate.max:
            result = sorted(results, key=lambda r: r.density)[::-1][0]
        elif aggregate == Aggregate.min:
            result = sorted(results, key=lambda r: r.density)[0]
        else:
            result_sum = reduce(
                lambda x, y: EMDDResult(
                    target=x.target + y.target,
                    scale=x.scale + y.scale,
                    density=x.density + y.density
                ), results
            )
            result = EMDDResult(
                target=result_sum.target / len(results),
                scale=result_sum.scale / len(results),
                density=result_sum.density / len(results)
            )

        print("Using result", result, "for prediction")

        for i in range(0, len(bags)):

            instances = []
            instance_probabilities = []
            probabilities = EMDD.positive_instance_probability(result.target, result.scale, bags[i].instances)
            for j in range(0, len(probabilities)):
                if probabilities[j] > threshold:
                    instances.append(j)
                    instance_probabilities.append(probabilities[j])

            if len(instances) > 0:
                #print("Bag", i, "is positive with instances", instances, "and probabilities", instance_probabilities)
                prediction_results.append(
                    PredictionResult(bag=i, instances=instances)
                )

        return prediction_results

    def train(self, threshold, perform_scaling=False, runs=10, k=5):
        results = []

        for bag in self.training_data.training_bags:
            for instance in bag.instances:
                instance.used_as_target = False

        for i in range(0, runs):
            run_results = []

            for j in range(0, k):
                random_instance = self.training_data.random_positive_training_bag().random_unused_instance()
                random_instance.used_as_target = True

                run_results.append(self.run(
                    threshold, perform_scaling, random_instance.features, numpy.ones(random_instance.features.size)
                ))

            best_result = sorted(run_results, key=lambda r: r.density)[0]
            print("Run", i, "Target:", best_result.target, "Scale:", best_result.scale, "Density", best_result.density)
            results.append(best_result)

        return results

    def run(self, threshold, perform_scaling, target, scale):
        density_difference = math.inf
        previous_density = math.inf
        density = 0

        while density_difference > threshold:

            optimal_instances = []
            for bag in self.training_data.training_bags:
                probabilities = EMDD.positive_instance_probability(target, scale, bag.instances)
                #print("run", run, "positive", bag.is_positive(), "bag", bag.index, "instance", numpy.argmax(probabilities), "probability", probabilities[numpy.argmax(probabilities)])
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
                jac=EMDD.diverse_density_gradient,
                x0=params,
                args=optimal_instances,
                bounds=bounds,
                method='L-BFGS-B',
                options={
                    'ftol': 1.0e-06,
                    'maxfun': 100000,
                    'maxiter': 2000,
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
            density_difference = (previous_density - density)
            previous_density = density

        return EMDDResult(target=target, scale=scale, density=density)

    @staticmethod
    def positive_instance_probability(target, scale, instances):
        x = numpy.tile(target, (len(instances), 1))
        s = numpy.tile(scale, (len(instances), 1))
        b_ij = numpy.array(list(map(lambda i: i.features, instances)))

        distances = numpy.mean(numpy.square(s) * numpy.square(b_ij - x), 1)

        return numpy.exp(numpy.negative(distances))

    @staticmethod
    def diverse_density(params, instances):
        num_features = instances[0].features.size
        scaling = params.size != num_features

        if not scaling:
            target = params
            scale = numpy.ones(num_features)
        else:
            target = params[0:num_features]
            scale = params[num_features:2 * num_features]

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

    @staticmethod
    def diverse_density_gradient(params, instances):
        num_features = instances[0].features.size
        scaling = params.size != num_features

        if not scaling:
            target = params
            scale = numpy.ones(num_features)
            gradient = numpy.zeros(num_features)
        else:
            target = params[0:num_features]
            scale = params[num_features:2 * num_features]
            gradient = numpy.zeros(2 * num_features)

        p = EMDD.positive_instance_probability(target, scale, instances)

        for d in range(0, num_features):
            for i in range(0, len(instances)):
                if instances[i].bag.is_positive():
                    if p[i] == 0:
                        p[i] = 1.0e-10

                    gradient[d] -= (2 / num_features) * \
                                   (scale[d] ** 2) * \
                                   (instances[i].features[d] - target[d])

                    if scaling:
                        gradient[d + num_features] += (2 / num_features) * scale[d] * \
                                                      ((instances[i].features[d] - target[d]) ** 2)
                else:
                    if p[i] == 1:
                        p[i] = 1 - 1.0e-10

                    gradient[d] += (1 / (1 - p[i])) * \
                                   (2 / num_features) * \
                                   (scale[d] ** 2) * \
                                   (instances[i].features[d] - target[d])

                    if scaling:
                        gradient[d + num_features] -= (1 / (1 - p[i])) * \
                                                      (2 / num_features) * scale[d] * \
                                                      ((instances[i].features[d] - target[d]) ** 2)

        return gradient


class MatlabTrainingData:

    def __init__(self, file_name, handler):
        data = handler(scipy.io.loadmat(file_name))

        self.training_bags = data["training_bags"]
        self.test_bags = data["test_bags"]

    def random_positive_training_bag(self):
        positive_training_bags = [bag for bag in self.training_bags if bag.is_positive() and bag.is_unused()]
        return positive_training_bags[random.randrange(0, len(positive_training_bags))]


class Bags(Sequence):

    def __init__(self, bags):
        self.bags = bags

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


def load_data(mat, bags_key, labels_key):
    bags = []

    mat_bags = mat[bags_key]
    it_bags = numpy.nditer(mat_bags, ["refs_ok", "c_index"])
    while not it_bags.finished:
        mat_bag = mat_bags[0][it_bags.index]
        label = mat[labels_key][0][it_bags.index]

        bag = Bag(it_bags.index, label)
        i = 0
        for instance in mat_bag:
            bag.add_instance(instance)
            inst_list = list(map(lambda x: str(x), instance.tolist()))
            #print("INST_{}_{},BAG_{},{},{}".format(i, it_bags.index, it_bags.index, label, ",".join(inst_list)))
            i += 1

        bags.append(bag)

        it_bags.iternext()

    return bags


def load_synth_data(mat):
    return {
        "training_bags": load_data(mat, "bag", "labels"),
        "test_bags": []
    }


def load_dr_data(mat):
    return {
        "training_bags": load_data(mat, "bag", "labels"),
        "test_bags": load_data(mat, "testBags", "testlabels"),
    }


def load_musk_data(mat):
    pp.pprint(mat)

    bag_map = {}
    it_bag_ids = numpy.nditer(mat["bag_ids"], ["refs_ok", "c_index"])
    while not it_bag_ids.finished:
        bag_index = mat["bag_ids"][0][it_bag_ids.index]
        if bag_index not in bag_map:
            bag_map[bag_index] = Bag(bag_index, -1)

        bag = bag_map[bag_index]
        bag.add_instance(numpy.array(mat["features"][it_bag_ids.index].A[0]))

        if not bag.is_positive() and mat["labels"].A[0][it_bag_ids.index] == 1:
            bag.label = 1

        it_bag_ids.iternext()

    return {
        "training_bags": list(bag_map.values()),
        "test_bags": list(bag_map.values())
    }


def load_animal_data(mat):
    pp.pprint(mat)
    bag_map = {}
    it_bag_ids = numpy.nditer(mat["bag_ids"], ["refs_ok", "c_index"])
    while not it_bag_ids.finished:
        bag_index = mat["bag_ids"][0][it_bag_ids.index]
        if bag_index not in bag_map:
            bag_map[bag_index] = Bag(bag_index, -1)

        bag = bag_map[bag_index]
        bag.add_instance(numpy.array(mat["features"][it_bag_ids.index].A[0]))

        if not bag.is_positive() and mat["labels"].A[0][it_bag_ids.index] == 1:
            bag.label = 1

        it_bag_ids.iternext()

    return {
        "training_bags": list(bag_map.values()),
        "test_bags": list(bag_map.values())
    }


def load_fake_data(mat):
    positive_instance = numpy.array([1, 0, 1, 1, 0])
    negative_instance = numpy.array([0, 1, 0, 0, 0])

    bags = []
    for i in range(0, 100):
        bag = Bag(i, i % 2)
        if i % 2 == 0:
            for j in range(0, 10):
                bag.add_instance(negative_instance)
        else:
            num_positive = random.randrange(0, 5) + 1
            for j in range(0, num_positive):
                bag.add_instance(positive_instance)

            for j in range(0, 10 - num_positive):
                bag.add_instance(negative_instance)

        bags.append(bag)

        print("bag", i, bag.is_positive(), "has instances")
        for instance in bag.instances:
            print("    instance", instance.features)

    return {
        "training_bags": bags,
        "test_bags": bags
    }

# Comment and uncomment as needed

#training_data = MatlabTrainingData('training-data/musk1norm_matlab.mat', load_fake_data) # for fake data
#training_data = MatlabTrainingData('training-data/musk1norm_matlab.mat', load_musk_data) # musk1
#training_data = MatlabTrainingData('training-data/musk2norm_matlab.mat', load_musk_data) # musk1
#training_data = MatlabTrainingData('training-data/synth_data_1.mat', load_synth_data) # synth data 1
#training_data = MatlabTrainingData('training-data/synth_data_4.mat', load_synth_data) # synth data 4
training_data = MatlabTrainingData('training-data/DR_data.mat', load_dr_data) # DR data
#training_data = MatlabTrainingData('training-data/elephant_100x100_matlab.mat', load_animal_data) # elephant
#training_data = MatlabTrainingData('training-data/fox_100x100_matlab.mat', load_animal_data) # fox
#training_data = MatlabTrainingData('training-data/tiger_100x100_matlab.mat', load_animal_data) # tiger

runs = 10

print(runs, "runs")

emdd = EMDD(training_data)
results = emdd.train(0.1, perform_scaling=True, runs=runs)

test_bags = training_data.training_bags

threshold = 0.5

prediction_results = EMDD.predict(results=results, bags=test_bags, threshold=threshold, aggregate=Aggregate.avg)

actual_positive_instances = len([bag for bag in test_bags if bag.is_positive()])

true_positives = 0
false_positives = 0

positive_bag_indexes = list(map(lambda x: x.index, [bag for bag in test_bags if bag.is_positive()]))
negative_bag_indexes = list(map(lambda x: x.index, [bag for bag in test_bags if not bag.is_positive()]))
predicted_positive_bag_indexes = []

for prediction_result in prediction_results:

    predicted_positive_bag_indexes.append(prediction_result.bag)
    if test_bags[prediction_result.bag].is_positive():
        true_positives += 1
    else:
        false_positives += 1

false_negatives = len([index for index in positive_bag_indexes if index not in predicted_positive_bag_indexes])
true_negatives = len([index for index in negative_bag_indexes if index not in predicted_positive_bag_indexes])

print("Threshold", threshold)
print("Total bags", len(test_bags))
print("Total positive bags", actual_positive_instances)
print("Total negative bags", len(negative_bag_indexes))
print("Total predicted positive bags", len(prediction_results))
print("Total predicted negative bags", true_negatives)
print("True positives", true_positives)
print("False positives", false_positives)
print("False negatives", false_negatives)
print("Accuracy", (true_positives + true_negatives) / len(test_bags))
print("Precision", true_positives / (true_positives + false_positives))
print("Recall", true_positives / (true_positives + false_negatives))

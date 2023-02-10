import numpy

def rbf(x, mu, beta):
    return numpy.exp(-(x - mu)**2 / beta**2)

def cutoff_func(r, d):
    return numpy.where(r > d, 0, 0.5 * (numpy.cos((numpy.pi * r) / d) + 1))

def even_samples(min_range, max_range, n_samples):
    samples = numpy.empty(n_samples)
    len_range = (max_range - min_range) / n_samples

    for i in range(0, n_samples):
        samples[i] = min_range + len_range * (i + 1)

    return samples

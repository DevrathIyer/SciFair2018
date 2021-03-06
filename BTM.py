"""
Bi-Term Topic Model (BTM) for very short texts.
Literature Reference:
  Xiaohui Yan, Jiafeng Guo, Yanyan Lan, and Xueqi Cheng:
  "A biterm topic model for short texts"
  In Proceedings of WWW '13, Rio de Janeiro, Brazil, pp. 1445-1456.
  ACM, DOI: https://doi.org/10.1145/2488388.2488514
This module requires pre-processing of textual data,
it only deals with integer term IDs, not strings.
BTModel is the plain topic model and can generate random data
according to its parametrization.
BTMGibbsSampler can infer a BTModel from data.
This module may run slow under CPython, consider execution with the
PyPy Python Interpreter to drastically speed up model inference.
"""

import random


def normalize(v):
    """L1-Normalize to obtain a vector of probabilities"""
    norm = sum(v)
    return [float(i) / norm for i in v]


def multinomial(p):
    """Sample multinomial-distributed index from Mul(p)"""
    r = random.random()
    for i in range(len(p)):
        r = r - p[i]
        if r < 0:
            return i
    return len(p) - 1


class BTModel(object):
    """Parameters of the bi-term topic model (BTM)"""

    def __init__(self, num_topics, num_words, topic_terms, topic_proportions):
        self.num_words = num_words
        self.num_topics = num_topics
        self.topic_terms = topic_terms  # topic_terms[z][w] = P( W=w | Z=z )
        self.topic_proportions = topic_proportions

    def sample_biterm(self):
        """Draw a single sample bi-term from this model"""
        z = multinomial(self.topic_proportions)
        w1 = multinomial(self.topic_terms[z])
        w2 = multinomial(self.topic_terms[z])
        return w1, w2

    def sample(self, n):
        """Draw a number of sample bi-terms from this model"""
        return [self.sample_biterm() for _ in range(n)]


class BTMGibbsSampler(object):
    """Inference of model parameters from bi-term samples.
    biterms: List of observed bi-terms (pairs of integer term IDs)
    num_topics: Number of desired topics to learn
    num_words: Number of words such that every term ID t
      is in 0 <= t < num_words.
    alpha: Dirichlet prior for global topic proportions,
    beta: Dirichlet prior for per-topic word distributions"""

    def __init__(self, biterms, num_topics, num_words, alpha, beta):
        self.biterms = biterms
        self.num_topics = num_topics
        self.num_words = num_words
        self.num_biterms = len(biterms)
        self.topic_terms = [[0] * num_words for _ in range(num_topics)]
        self.topic_sums = [0] * num_topics
        self.biterm_topics = [0] * len(biterms)
        self.alpha = alpha
        self.beta = beta
        self.randomize()

    def fit(self, steps):
        """Infer the model, higher steps yield a less random model."""
        for i in range(steps):
            self.update()

    def as_model(self):
        """Obtain the model once fit() has been called"""
        k_alpha = self.num_biterms + self.num_topics * self.alpha
        m_beta = self.num_words * self.beta
        n_topics = [self.topic_sums[z] * 2 for z in range(self.num_topics)]
        topic_terms = [normalize([
            (self.topic_terms[z][w] + self.beta) / (n_topics[z] + m_beta)
            for w in range(self.num_words)])
            for z in range(self.num_topics)]
        topic_proportions = normalize([
            (self.topic_sums[z] + self.alpha) / (self.num_biterms + k_alpha)
            for z in range(self.num_topics)
        ])
        return BTModel(self.num_topics, self.num_words,
                       topic_terms, topic_proportions)

    def randomize(self):
        """Bootstrap a random model to be optimized by fit()"""
        for i, bt in enumerate(self.biterms):
            w1, w2 = bt
            z = random.randint(0, self.num_topics - 1)
            self.biterm_topics[i] = z
            self.topic_terms[z][w1] += 1
            self.topic_terms[z][w2] += 1
            self.topic_sums[z] += 1

    def update(self):
        """Single optimization step"""
        for index, bt in enumerate(self.biterms):
            w1, w2 = bt

            old_topic = self.biterm_topics[index]

            # remove current assignment from statistics
            self.topic_terms[old_topic][w1] -= 1
            self.topic_terms[old_topic][w2] -= 1
            self.topic_sums[old_topic] -= 1
            m_beta = self.num_words * self.beta

            # estimate Gibbs sampling distribution for topic
            posterior = normalize([
                (self.topic_sums[z] + self.alpha)
                * (self.topic_terms[z][w1] + self.beta)
                / (self.topic_sums[z] * 2 + m_beta)
                * (self.topic_terms[z][w2] + self.beta)
                / (self.topic_sums[z] * 2 + m_beta)
                for z in range(self.num_topics)])

            # draw new topic
            new_topic = multinomial(posterior)

            # re-assign new topic and update statistics
            self.biterm_topics[index] = new_topic
            self.topic_terms[new_topic][w1] += 1
            self.topic_terms[new_topic][w2] += 1
            self.topic_sums[new_topic] += 1


def docs_to_biterms(docs):
    """Converts non-binary documents to bi-terms by producing all distinct pairs"""
    biterms = []
    for d in docs:
        for i1, t1 in enumerate(d):
            for i2, t2 in enumerate(d):
                if i1 != i2 and t1 != t2:
                    biterms.append((t1, t2))
    return biterms


if __name__ == '__main__':

    # This testing routine first generates 10 topics (5 rows, 5 columns)
    # and fixed topic proportions, then samples a number of bi-terms
    # and feeds them to the Gibbs sampler to infer a model
    # only from the data. Visual comparison is provided by
    # printing original and inferred topics as prob. distributions.

    # Example:
    # This topic has been generated:
    #  0.0  0.0  0.0  0.0  0.0
    # 20.0 20.0 20.0 20.0 20.0  <- single row
    #  0.0  0.0  0.0  0.0  0.0
    #  0.0  0.0  0.0  0.0  0.0
    #  0.0  0.0  0.0  0.0  0.0

    # And should re-appear in the inferred model with slight random noise:
    #  0.0  0.0  0.0  0.0  0.0
    # 20.0 19.9 20.8 20.1 19.3  <- inferred topic
    #  0.0  0.0  0.0  0.0  0.0
    #  0.0  0.0  0.0  0.0  0.0
    #  0.0  0.0  0.0  0.0  0.0

    # note that topics are randomly permuted after inference.

    def example_topics(n=5):
        """Generate 2*n topics of n*n words"""
        topics = []
        for i in xrange(n):
            v = normalize([1.0 if j % n == i else 0.0 for j in xrange(n*n)])
            h = normalize([1.0 if j / n == i else 0.0 for j in xrange(n*n)])
            topics.append(v)
            topics.append(h)

        return topics

    def print_topic(t):
        n = int(len(t) ** 0.5)
        assert n * n == len(t)
        for i in xrange(n):
            for j in xrange(n):
                print '%4.1f' % (100 * t[i * n + j]),
            print

    tops = example_topics(5)
    props = normalize(range(1, 11))

    print 'proportions:', [round(p, 2) for p in props]

    for t in tops:
        print_topic(t)
        print '-' * 40

    num_words = 25
    num_pairs = 100000

    model = BTModel(10, num_words, tops, props)
    data = model.sample(num_pairs)

    print ' Example bi-terms:', data[:20], data[-20:]

    print 'Running BTM Gibbs sampler...'
    sampler = BTMGibbsSampler(data, 10, num_words, .01, .01)
    sampler.fit(500)

    rmodel = sampler.as_model()

    print '=' * 40
    print 'proportions:', [round(p, 2) for p in rmodel.topic_proportions]
    print 'sorted:', [round(p, 2) for p in sorted(rmodel.topic_proportions)]
    for t in rmodel.topic_terms:
        print_topic(t)
        print '-' * 40
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score, average_precision_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import joblib
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
from num2words import num2words
import re, string
import tensorflow_hub as hub
from sklearn.manifold import TSNE
import networkx as nx
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import textstat
from spellchecker import SpellChecker
import random
from skimage.feature import graycomatrix, graycoprops
import cv2 as cv
import os
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from scipy.stats import ttest_ind
from scipy.stats import wasserstein_distance_nd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

"""
A class for turning pandas dataframes filled with categorical features and mapping them to integers

feature_names: The names of the features being mapped to integers
feature_mappings: The mappings for each feature in feature_names represented as a list of dictionaries. The dictionaries have a each structure of {unmapped_feature: mapped_feature}

__init__: Initializes the class to prepare for fitting
fit: Sets up all mappings based on the provided data. If feature_names was not given during initialization, also sets feature_names
transform: Maps provided data to integers using feature_mappings
fit_transform: Runs the fit and transform function on the passed data
"""
class Categorizer():
  """
  Initializes the categorizer to be ready for fitting

  feature_names: The names of the features to be mapped. If None, the names of the features to be mapped will be determined by the column names of the data passed to fit
  """
  def __init__(self, feature_names=None):
    self.feature_names = feature_names
    self.feature_mappings = []
  """
  Creates mappings based on the passed dataframe. If feature_names has not already been set, then it also sets feature_names to be the columns of the passed dataframe

  df: The dataframe containing the data to create mappings for
  """
  def fit(self, df):
    if (self.feature_names == None):
      self.feature_names = df.columns

    categorical_df = df[self.feature_names]

    self.feature_mappings = []
    for feature_name in self.feature_names:
      unique_values = categorical_df[feature_name].unique()
      new_mapping = {unique_values[i]: i for i in range(len(unique_values))}
      self.feature_mappings.append(new_mapping)
  """
  Transforms the categorical data within a dataframe to integers based on feature_mappings

  df: The dataframe to transform

  returns: A pandas dataframe with the columns column_names and with values that are mapping of df
  """
  def transform(self, df):
    categorical_df = df[self.feature_names]

    transformed_df = pd.DataFrame()
    for feature_i in range(len(self.feature_names)):
      current_feature_name = self.feature_names[feature_i]
      current_mapping = self.feature_mappings[feature_i]

      transformed_df[current_feature_name] = categorical_df[current_feature_name].map(current_mapping).astype(float)

    return transformed_df
  """
  Fits and transforms provided data, combining fit and transform functions

  df: A pandas dataframe containing the data to fit to and transform

  returns: A pandas dataframe with the columns column_names and with values that are mapping of df
  """
  def fit_transform(self, df):
    self.fit(df)
    transformed_df = self.transform(df)
    return transformed_df

"""
A base class for sampling data from a simulated distribution. If instantiated, draws random datapoints from the data fit to

data: The data the simulated distribution will draw from
n_datapoints: The number of datapoints the simulated distribution can draw from

init: Initializes the simulated distribution to be fit
fit: Sets the data the simulated distribution should draw from
sample: Pulls a given number of random samples from the simulated distribution
"""
class Distribution():
    """
    Initializes the simulated distribution to be fit
    """
    def __init__(self):
      self.data = None
      self.n_datapoints = 0
    """
    Sets the data the simulated distribution should draw from

    data: The datapoints to sample from in the future
    """
    def fit(self, data):
      self.data = data
      self.n_datapoints = data.shape[0]
    """
   Pulls a given number of random samples from the simulated distribution

    n_samples: The number of samples to pull

    returns: Given number of datapoints from the simulated distribution
    """
    def sample(self, n_samples=1):
      return self.data[torch.randints(self.n_datapoints, (n_samples,))]

"""
A class for simulating a numeric distribution to any finite degree of precision by imagining the distribution as a set of uniform distributions determined with equal frequency buckets

precision_degree: The number of buckets the data is divided into, and the number of uniform distributions the simulated distribution is made up of
bucket_means: The mean value of each bucket
bucket_lefts: The minimum value of each bucket, and the left edge of each uniform distribution
bucket_rights: The minimum value of the next largest bucket of each bucket, except for the rightmost bucket, where it is the maximum value in that bucket. Also the maximum value in each uniform distribution

init: Initializes the simulated distribution to be ready for fitting
fit: Divides given data into buckets to be drawn from later
sample: Draws a given number of samples from the simulated distribution
set_precision_degree: Sets the precision degree to a new value. Also undoes any fitting already done.
cdf: The cumulative density function of the simulated distribution
icdf: The inverse cumulative density function of the simulated distribution
drift_mean: Drifts the mean of the simulated distribution
drift_variance: Drifts the variance of the simulated distribution
drift_skew: Drifts the skew of the simulated distribution
transform_distribution: Transforms the simulated distribution according to passed function that takes in all bucket lefts, means, and rights at once
transform_elements: Transforms the simulated distribution according to passed function that takes in each bucket left, bucket mean, and bucket right
"""
class NumericDistribution(Distribution):
    """
    Initializes the simulated distribution to be ready for fitting

    precision_degree: The number of buckets the distribution should be divided into
    """
    def __init__(self, precision_degree=None):
        self.precision_degree = precision_degree

        if (self.precision_degree != None):
          self.bucket_means = torch.zeros((self.precision_degree,))
          self.bucket_lefts = torch.zeros((self.precision_degree,))
          self.bucket_rights = torch.zeros((self.precision_degree,))

    """
    Fits the simulated distribution to the data provided

    data: The data to fit the simulated distribution to as a 1D pytorch tensor.
    """
    def fit(self, data):
        if (self.precision_degree == None):
          self.precision_degree = data.shape[0]

          self.bucket_means = torch.zeros((self.precision_degree,))
          self.bucket_lefts = torch.zeros((self.precision_degree,))
          self.bucket_rights = torch.zeros((self.precision_degree,))

        sorted_data, _ = torch.sort(data)

        n_samples = sorted_data.shape[0]
        step_size = n_samples / self.precision_degree

        for bucket_i in range(self.precision_degree):
            left_bucket_i, right_bucket_i = int(bucket_i * step_size), int((bucket_i + 1) * step_size)
            current_bucket = sorted_data[left_bucket_i:right_bucket_i]

            self.bucket_means[bucket_i] = torch.mean(current_bucket)
            self.bucket_lefts[bucket_i] = current_bucket[0]
            self.bucket_rights[bucket_i] = sorted_data[min(right_bucket_i, sorted_data.shape[0] - 1)]

    """
    Generates new samples from the simulated distribution

    n_samples: The number of samples to be generated

    returns: given number of samples from the distribution in a 1D pytorch tensor
    """
    def sample(self, n_samples=1):
        #random_nums = torch.rand((n_samples,))
        #return self.icdf(random_nums)

        bucket_indices = torch.randint(self.precision_degree, (n_samples,))
        sample_magnitudes = torch.rand((n_samples,))

        sample_mins, sample_maxes = self.bucket_lefts[bucket_indices], self.bucket_rights[bucket_indices]
        samples = sample_mins + sample_magnitudes * (sample_maxes - sample_mins)
        return samples

    """
    The cumulative density function of the simulated distribution

    values: The values to be passed into the cumulative density function as a 1D pytorch tensor

    returns: The portion of the simulated distribution less than or equal to each value as a 1D pytorch tensor
    """
    def cdf(self, values):
        value_bucket_indices = torch.tensor([torch.nonzero(values[i] > self.bucket_lefts)[-1] for i in range(values.shape[0])])

        value_bucket_lefts = self.bucket_lefts[value_bucket_indices]
        value_bucket_rights = self.bucket_rights[value_bucket_indices]

        cdf_values = (value_bucket_indices.float() + (values - value_bucket_lefts) / (value_bucket_rights - value_bucket_lefts)) / self.precision_degree

        return cdf_values
    """
    The inverse cumulative density function of the simulated distribution

    values: Values in range range [0-1] representing portions of the simulated distribution to be passed into the inverse cumulative density function as a 1D pytorch tensor

    returns: The values in the simulated distribution such that they have a portion of the simulated distribution less than or equal themselves equal to the passed values
    """
    def icdf(self, values):
        value_bucket_indices = torch.floor(values * self.precision_degree).long()
        value_bucket_magnitudes = (values % (1 / self.precision_degree)) * self.precision_degree

        icdf_mins, icdf_maxes = self.bucket_lefts[value_bucket_indices], self.bucket_rights[value_bucket_indices]
        icdf_values = icdf_mins + value_bucket_magnitudes * (icdf_maxes - icdf_mins)
        return icdf_values

    """
    Sets the degree of precision for the model to a new value. Also resets bucket means, lefts, and rights

    new_precision_degree: The new degree of precision to use
    """
    def set_precision_degree(self, new_precision_degree):
        self.precision_degree = new_precision_degree

        self.bucket_means = torch.zeros((self.precision_degree,))
        self.bucket_lefts = torch.zeros((self.precision_degree,))
        self.bucket_rights = torch.zeros((self.precision_degree,))

    """
    Drifts the mean of the simulated distribution, affecting the means, lefts, and rights of all buckets

    alpha: The amount to shift the mean by
    """
    def drift_mean(self, alpha):
        self.bucket_means = self.bucket_means + alpha
        self.bucket_lefts = self.bucket_lefts + alpha
        self.bucket_rights = self.bucket_rights + alpha
    """
    Drifts the variance of the simulated distribution, affecting the means, lefts, and rights of all buckets

    alpha: The percentage to shift the variance by
    """
    def drift_variance(self, alpha):
        overall_mean = torch.mean(self.bucket_means)
        self.bucket_means = (1 + alpha) * (self.bucket_means - overall_mean) + overall_mean
        self.bucket_lefts = (1 + alpha) * (self.bucket_lefts - overall_mean) + overall_mean
        self.bucket_rights = (1 + alpha) * (self.bucket_rights - overall_mean) + overall_mean
    """
    Drifts the skew of the simulated distribution, affecting the means, lefts, and rights of all buckets

    alpha: The value to shift the skew by
    """
    def drift_skew(self, alpha):
        self.bucket_means = torch.sign(self.bucket_means) * torch.pow(self.bucket_means, 1 + (alpha / 2))

    """
    Transforms the simulated distribution as a whole according to the given function

    f: The function to transform the distribution. Takes the bucket means, bucket lefts, and bucket rights of the simulated distribution as input, and should return the transformed bukcet means, bucket lefts, and bucket rights
    """
    def transform_distribution(self, f):
        self.bucket_means, self.bucket_lefts, self.bucket_rights = f(self.bucket_means, self.bucket_lefts, self.bucket_rights)
    """
    Transforms the simulated distribution by transforming each element according to the given function

    f: The function to transform the distribution. Takes in a bucket mean, bucket left, and bucket right as input, and should return a transformed bucket mean, bucket left, and bucket right
    """
    def transform_elements(self, f):
        for bucket_i in range(self.precision_degree):
            self.bucket_means[bucket_i], self.bucket_lefts[bucket_i], self.bucket_rights[bucket_i] = f(self.bucket_means[bucket_i], self.bucket_lefts[bucket_i], self.bucket_rights[bucket_i])
            
"""
A class for simulating a categorical distribution to be sampled from

n_categories: The number of categories in the distribution
category_portions: The portion of the simulated distribution that each category makes up stored as a 1D pytorch tensor
thresholds: Cumulative sums of category_portions. A cumulative version of the simulated distribution

init: Initializes the simulated distribution to be ready for fitting
fit: Fits the simulated distribution to provided data
sample: Pulls a given number of samples from the simulated distribution
"""
class CategoricalDistribution(Distribution):
    """
    Initializes the simulated distribution to be ready for fitting
    """
    def __init__(self):
        self.n_categories = 0
        self.category_portions = torch.zeros((0,))
        self.thresholds = torch.zeros((0,))

    """
    Fits the simulated distribution to provided data

    data: The data to fit the simulated distribution to represented as a 1D pytorch tensor of integers, where each unique integer represents a category starting at 0
    """
    def fit(self, data):
        self.n_categories = int(torch.max(data).item()) + 1
        n_samples = data.shape[0]

        self.category_portions = torch.zeros((self.n_categories,))
        for category_i in range(self.n_categories):
            self.category_portions[category_i] = torch.sum(data == category_i) / n_samples

        self.thresholds = torch.zeros((self.n_categories,))
        for category_i in range(self.n_categories):
            self.thresholds[category_i:] += self.category_portions[category_i]

    """
    Pulls a given number of samples from the simulated distribution

    n_samples: The number of samples to pull from the simulated distribution

    returns: The given number of samples as a 1D pytorch tensor filled with random integers in proportion to the categories in the simulated distribution
    """
    def sample(self, n_samples=1):
        random_nums = torch.rand((n_samples,))
        samples = torch.zeros((n_samples,))

        adjusted_thresholds = torch.cat((torch.tensor([0]), self.thresholds))

        for category_i in range(self.n_categories):
            current_min_threshold = adjusted_thresholds[category_i]
            current_max_threshold = adjusted_thresholds[category_i + 1]

            samples = torch.where(torch.logical_and(random_nums >= current_min_threshold, random_nums < current_max_threshold), torch.full((n_samples,), category_i), samples)

        return samples
"""
A CategoricalDistribution that adds cumulative density function and inverse cumulative density function functionality

__init__: Sets up the distribution for fitting by calling __init__ for CategoricalDistribution
cdf: Acts as a cumulative density function of the categorical distribution, imagining the distribution as having each category at an integer mark
icdf: Acts as an inverse cumulative density function of the categorical distribution, imagining the distribution as having each category at an integer mark
"""
class PseudoCategoricalDistribution(CategoricalDistribution):
  """
  Initializes the distribution such that it is ready to be fit.
  """
  def __init__(self):
    super().__init__()
  """
  Acts as a cumulative density function for the distribution. While this isn't normally definted for categorical distributions, this function acts as if all values of a given category are at different integer marks

  values: The values to be passed into the cumulative density function as a 1D pytorch tensor

  returns: The portion of the simulated distribution less than or equal to each value as a 1D pytorch tensor
  """
  def cdf(self, values):
    n_values = values.shape[0]
    floored_values = torch.floor(values)

    in_distribution_indices = torch.logical_and(floored_values >= 0, floored_values <= (self.n_categories - 1))
    in_distribution_floored_values = floored_values[in_distribution_indices]

    cdf_values = torch.zeros((n_values,))

    cdf_values[in_distribution_indices] = self.thresholds[in_distribution_floored_values]

    cdf_values = torch.where(floored_values < 0, torch.zeros((n_values,)), cdf_values)
    cdf_values = torch.where(floored_values > (self.n_categories - 1), torch.ones((n_values,)), cdf_values)

    return cdf_values
  """
  Acts as an inverse cumulative density function for the distribution. While this isn't normally definted for categorical distributions, this function acts as if all values of a given category are at different integer marks

  values: Values in range range [0-1] representing portions of the simulated distribution to be passed into the inverse cumulative density function as a 1D pytorch tensor

  returns: The values in the simulated distribution such that they have a portion of the simulated distribution less than or equal themselves equal to the passed values
  """
  def icdf(self, values):
    n_values = values.shape[0]
    adjusted_thresholds = torch.cat((torch.tensor([0]), self.thresholds))

    samples = torch.zeros((n_samples,))
    for category_i in range(self.n_categories):
      current_min_threshold = adjusted_thresholds[category_i]
      current_max_threshold = adjusted_thresholds[category_i + 1]

      samples = torch.where(torch.logical_and(values >= current_min_threshold, values < current_max_threshold), torch.full((n_samples,), category_i), samples)

    return samples

"""
The cdf function for the standard normal distribution (mean=0, std=1)

values: Values from the standard normal distribution as a 1D pytorch tensor

returns: The portion of the standard normal distribution before and including each value as a 1D pytorch tensor
"""
def normal_cdf(values):
  return 0.5 * (1 + torch.erf(values / math.sqrt(2)))
"""
Transforms a pseudo correlation matrix into a correlation matrix

pseudo_correlation_matrix: The pseudo-correlation matrix to be transformed into a correlation matrix as a 2D pytorch tensor

returns: pseudo_correlation_matrix transformed into a correlation matrix
"""
def fix_pseudo_correlation_matrix(pseudo_correlation_matrix):
  n_features, _ = pseudo_correlation_matrix.shape

  #Make the matrix symmetric
  symmetric_correlation_matrix = (pseudo_correlation_matrix + pseudo_correlation_matrix.T) / 2

  #Make the matrix positive semidefinite
  eigen_values = torch.real(torch.linalg.eigvals(symmetric_correlation_matrix))
  min_eigen_value = torch.min(eigen_values)

  upscaled_correlation_matrix = symmetric_correlation_matrix + torch.diag(torch.full((n_features,), max(-1 * min_eigen_value, 0)))

  #Normalize matrix so diagonals are 1
  scaling_matrix = torch.sqrt(torch.diag(1 / torch.diag(upscaled_correlation_matrix)))
  final_correlation_matrix = scaling_matrix @ upscaled_correlation_matrix @ scaling_matrix

  return final_correlation_matrix

"""
The inverse cdf function for the standard normal distribution (mean=0, std=1)

values: Values in the range [0, 1] as a 1D pytorch tensor

returns: The values in the standard normal distribution such that they have a portion of the standard normal distribution less than or equal themselves equal to the passed values
"""
def normal_inverse_cdf(values):
  return torch.erfinv(2 * values - 1) * math.sqrt(2)
"""
A class for creating a gaussian copula to sample from, attempting to maintain relationships in mutlivariate data

correlation_matrix: The correlation matrix for the features the copula is fit to
decomposition_matrix: The cholesky decomposition of the correlation matrix
cdfs: The cumulative density functions of the simulated distributions for each feature the copula is fit to
n_features: The number of features in the copula

init: Initializes the copula to be ready for fitting
fit: Fits the copula to the data given
sample: Generates samples form the simulated distribution of data
copula_subset: Gives a copula based on a subset of features in the current copula
set_correlation_matrix: Sets correlation_matrix and decomposition_matrix according to given correlation matrix
"""
class GaussianCopula(Distribution):
  """
  Initializes the copula to be ready for fitting
  """
  def __init__(self, precision_degree=None):
    self.correlation_matrix = torch.zeros((0, 0))
    self.decomposition_matrix = torch.zeros((0, 0))
    self.cdfs = []
    self.n_features = 0
    self.precision_degree = precision_degree
    self.features_are_numeric = []
  """
  Fits the copula to the given data

  data: The data to fit the copula to as a 2D pytorch tensor where the rows are individual samples and the columns are all samples of a given feature
  """
  def fit(self, data, features_are_numeric=None):
    n_samples, n_features = data.shape

    if (features_are_numeric == None):
      features_are_numeric = torch.full((n_features,), True)
    self.features_are_numeric = features_are_numeric

    sorted_indices = torch.argsort(data, dim=0)

    feature_ranks = torch.zeros(data.shape)
    for i in range(n_features):
      feature_ranks[:, i][sorted_indices[:, i]] = torch.arange(n_samples).float()

    normalized_feature_ranks = feature_ranks / n_samples
    correlation_matrix = torch.corrcoef(normalized_feature_ranks.T)
    cholesky_decomposition = torch.linalg.cholesky(correlation_matrix)

    self.correlation_matrix = correlation_matrix
    self.decomposition_matrix = cholesky_decomposition

    self.cdfs = []
    for i in range(n_features):
      if (self.features_are_numeric[i]):
        new_cdf = NumericDistribution(precision_degree=self.precision_degree)
      else:
        new_cdf = PseudoCategoricalDistribution()
      new_cdf.fit(data[:, i])
      self.cdfs.append(new_cdf)

    self.n_features = n_features
  """
  Generates new samples from the simulated distribution using the copula

  n_samples: The number of samples desired to be generated

  returns: Generated samples from the simulated distribution as a 2D pytorch tensor where each row is a sample and each column is all samples of a feature
  """
  def sample(self, n_samples=1):
    #random_nums = torch.rand((n_samples, self.n_features))
    #icdf_values = normal_inverse_cdf(random_nums)
    icdf_values = torch.randn((n_samples, self.n_features))
    correlated_icdf_values = torch.stack([self.decomposition_matrix @ icdf_values[i, :] for i in range(n_samples)])
    correlated_random_nums = normal_cdf(correlated_icdf_values)
    samples = torch.stack([self.cdfs[i].icdf(correlated_random_nums[:, i]) for i in range(self.n_features)], dim=1)

    #TODO: Is this part neccesary?
    for i in range(self.n_features):
      if (not self.features_are_numeric[i]):
        samples[:, i] = torch.round(samples[:, i])

    return samples

  """
  Gives a copula based on a subset of features in the current copula

  subset_indices: The indices of the features that will make up the subset

  returns: A GaussianCopula based on the subset of feature defined by subset_indices
  """
  def copula_subset(self, subset_indices):
    subset_correlation_matrix = torch.stack([self.correlation_matrix[i, subset_indices] for i in subset_indices])
    subset_decomposition_matrix = torch.stack([self.decomposition_matrix[i, subset_indices] for i in subset_indices])
    subset_cdfs = [self.cdfs[i] for i in subset_indices]

    subset_copula = GaussianCopula(self.precision_degree)
    subset_copula.correlation_matrix = subset_correlation_matrix


    subset_data = self.data[:, subset_indices]
    subset_copula = GaussianCopula(precision_degree=self.precision_degree)
    subset_copula.fit(subset_data)

    return subset_copula

  """
  Sets correlation_matrix and decomposition_matrix according to given correlation matrix

  new_correlation_matrix: The correlation matrix to set correlation_matrix to
  """
  def set_correlation_matrix(self, new_correlation_matrix):
    cholesky_decomposition = torch.linalg.cholesky(new_correlation_matrix)

    self.correlation_matrix = new_correlation_matrix
    self.decomposition_matrix = cholesky_decomposition

"""
A class for creating an emprical copula to sample from and compare with each other

n_datapoints: The number of datapoints the copula is fit to
n_features: The number of features in the distribution
data: The raw data the copula is fit to sorted by the copula values of the datapoints in ascending order
copula_data: The copula normalized data the copula is fit to sorted by the copula values of the datapoints in ascending order
copula_values: The copula values of each datapoint in ascending order
precision_degree: The degree of precision the NumericDistributions representing each feature
cdfs: NumericDistributions of the each of the features in the distribution
features_are_numeric:

init: Initializes the copula for fitting
fit: Fits the copula to given data
sample: Samples a given number of datapoints from the copula
copula_subset:
"""
class EmpiricalCopula(Distribution):
  """
  Initializes the copula to be ready for fitting
  """
  def __init__(self, precision_degree=None):
    self.n_datapoints = 0
    self.n_features = 0
    self.data = torch.zeros((0, 0))
    self.copula_data = torch.zeros((0, 0))
    self.copula_values = torch.zeros((0,))
    self.precision_degree = precision_degree
    self.cdfs = []
    self.features_are_numeric = []
  """
  Fits the copula to given data

  data: The data to fit the copula to represented as a 2D pytorch tensor where each row is a datapoint in the distribution and each column is a feature in the distribution
  """
  def fit(self, data, features_are_numeric=None):
    n_samples, n_features = data.shape

    if (features_are_numeric == None):
      features_are_numeric = torch.full((n_features,), True)
    self.features_are_numeric = features_are_numeric
    if (self.precision_degree > (n_samples - 1)):
        self.precision_degree = n_samples - 1

    sorted_indices = torch.argsort(data, dim=0)

    feature_ranks = torch.zeros(data.shape)
    for i in range(n_features):
      feature_ranks[:, i][sorted_indices[:, i]] = torch.arange(n_samples).float()

    normalized_feature_ranks = feature_ranks / n_samples

    copula_values = torch.zeros((n_samples,))
    for i in range(n_samples):
      copula_values[i] = torch.sum(torch.all(normalized_feature_ranks <= normalized_feature_ranks[i, :], dim=1)) / n_samples

    sorted_copula_indices = torch.argsort(copula_values)
    sorted_copula_values = copula_values[sorted_copula_indices]
    sorted_normalized_feature_ranks = normalized_feature_ranks[sorted_copula_indices]
    sorted_data = data[sorted_copula_indices]

    self.data = sorted_data
    self.copula_data = sorted_normalized_feature_ranks
    self.copula_values = sorted_copula_values

    self.cdfs = []
    for i in range(n_features):
      if (self.features_are_numeric[i]):
        new_cdf = NumericDistribution(precision_degree=self.precision_degree)
      else:
        new_cdf = PseudoCategoricalDistribution()
      new_cdf.fit(data[:, i])
      self.cdfs.append(new_cdf)

    self.n_datapoints = n_samples
    self.n_features = n_features
  """
  Samples a given number of datapoints from the copula

  n_samples: The number of samples to pull from the simulated distribution

  returns: Generated datapoints from the simulated distribution represented as a 2D pytorch tensor where each row is a sampled datapoint and each column is a feature in the distribution
  """
  def sample(self, n_samples=1):
    #random_nums = torch.rand((n_samples,))

    #base_datapoint_indices = torch.floor(random_nums * (self.n_datapoints)).long()
    #copula_points = self.copula_data[base_datapoint_indices]
    #sampled_points = torch.stack([self.cdfs[i].icdf(copula_points[:, i]) for i in range(self.n_features)], dim=1)

    #return sampled_points
    #=================
    random_nums = torch.rand((n_samples,))

    base_datapoint_indices = torch.floor(random_nums * (self.n_datapoints - 1)).long()

    datapoint_starts = self.copula_data[base_datapoint_indices]
    datapoint_ends = self.copula_data[base_datapoint_indices + 1]
    datapoint_magnitudes = (base_datapoint_indices % (1 / (self.n_datapoints - 1))) / (1 / (self.n_datapoints - 1))

    copula_points = datapoint_starts + datapoint_magnitudes[:, None] * (datapoint_ends - datapoint_starts)
    sampled_points = torch.stack([self.cdfs[i].icdf(copula_points[:, i]) for i in range(self.n_features)], dim=1)

    for i in range(self.n_features):
      if (not self.features_are_numeric[i]):
        sampled_points[:, i] = torch.round(sampled_points[:, i])

    return sampled_points

  """
  Gives a copula based on a subset of features in the current copula

  subset_indices: The indices of the features that will make up the subset

  returns: An EmpiricalCopula based on the subset of feature defined by subset_indices
  """
  def copula_subset(self, subset_indices):
    subset_data = self.data[:, subset_indices]
    subset_numerical_features = self.features_are_numeric[subset_indices]
    subset_copula = EmpiricalCopula(precision_degree=self.precision_degree)
    subset_copula.fit(subset_data, features_are_numeric=subset_numerical_features)

    return subset_copula
"""
A class for creating a gaussian copula to sample from, attempting to maintain relationships in mutlivariate data

n_datapoints: The number of datapoints the copula is fit to
n_features: The number of features in the distribution
data: The raw data the copula is fit to sorted by the copula values of the datapoints in ascending order
copula_data: The copula normalized data the copula is fit to sorted by the copula values of the datapoints in ascending order
copula_values: The copula values of each datapoint in ascending order
precision_degree: The degree of precision the NumericDistributions representing each feature
cdfs: NumericDistributions of the each of the features in the distribution

init: Initializes the copula for fitting
fit: Fits the copula to given data
sample: Samples a given number of datapoints from the copula
copula_subset:
"""
class MiniEmpiricalCopula(Distribution):
  """
  Initializes the copula to be ready for fitting
  """
  def __init__(self):
    self.n_datapoints = 0
    self.n_features = 0
    self.data = torch.zeros((0, 0))
    self.copula_data = torch.zeros((0, 0))
  """
  Fits the copula to given data

  data: The data to fit the copula to represented as a 2D pytorch tensor where each row is a datapoint in the distribution and each column is a feature in the distribution
  """
  def fit(self, data):
    n_samples, n_features = data.shape

    sorted_indices = torch.argsort(data, dim=0)

    feature_ranks = torch.zeros(data.shape)
    for i in range(n_features):
      feature_ranks[:, i][sorted_indices[:, i]] = torch.arange(n_samples).float()

    normalized_feature_ranks = feature_ranks / n_samples

    self.data = data
    self.copula_data = normalized_feature_ranks

    self.n_datapoints = n_samples
    self.n_features = n_features
  """
  Samples a given number of datapoints from the copula

  n_samples: The number of samples to pull from the simulated distribution

  returns: Generated datapoints from the simulated distribution represented as a 2D pytorch tensor where each row is a sampled datapoint and each column is a feature in the distribution
  """
  def sample(self, n_samples=1):
    random_indices = torch.randint(self.n_datapoints, (n_samples,))
    sampled_points = self.data[random_indices]

    return sampled_points

  """
  Gives a copula based on a subset of features in the current copula

  subset_indices: The indices of the features that will make up the subset

  returns: An EmpiricalCopula based on the subset of feature defined by subset_indices
  """
  def copula_subset(self, subset_indices):
    subset_data = self.data[:, subset_indices]
    subset_copula_data = self.copula_data[:, subset_indices]

    subset_copula = MiniEmpiricalCopula()
    subset_copula.n_datapoints = self.n_datapoints
    subset_copula.n_features = self.n_features
    subset_copula.data = self.data[:, subset_indices]
    subset_copula.copula_data = self.copula_data[:, subset_indices]

    return subset_copula

sinkhorn_distance = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
"""
Calculates the distance between two EmpiricalCopulas

first_copula: The first copula to calculate distance between
second_copula: The second copula to calculate distance between

returns: THe sinkhorn distance between two copulas represented as a one element pytorch tensor
"""
def copula_distance(first_copula, second_copula):
  first_copula_distribution = first_copula.copula_data
  second_copula_distribution = second_copula.copula_data
  distance = sinkhorn_distance(first_copula_distribution, second_copula_distribution)
  return distance

"""
Processes a dataframe by turning categorical features into integers and removing any features that are non-numeric and non-categorical

df: The pandas dataframe to process
categorizer: The Categorizer to use for mapping categorical features to integers

returns: A pandas dataframe containing the numeric features of df and the categorical features of df mapped to integers according to categorizer
"""
def process_df(df, categorizer, NUMERIC_FEATURES=None, CATEGORICAL_FEATURES=None):
  numeric_df = df[NUMERIC_FEATURES]
  categorical_df = df[CATEGORICAL_FEATURES]

  mapped_categorical_df = categorizer.transform(categorical_df)

  processed_df = pd.concat((numeric_df, mapped_categorical_df), axis=1)
  return processed_df

"""
Takes a processed dataframe and turns it into a corresponding pytorch tensor

df: The processed pandas dataframe to turn into a pytorch tensor

returns: df represented as a 2D pytorch tensor
"""
def transform_processed_df_to_pytorch(df):
  data_tensor = torch.tensor(df.to_numpy())
  return data_tensor
"""
Takes an unprocessed dataframe and turns it into a processed dataframe

df: The pandas dataframe to process
categorizer: The Categorizer to use while processing the categorical features in df

returns: The processed version of df
"""
def transform_unprocessed_df_to_processed_df(df, categorizer, NUMERIC_FEATURES=None, CATEGORICAL_FEATURES=None):
  processed_df = process_df(df, categorizer, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
  return processed_df

"""
Takes an unprocessed dataframe and turns it into a pytorch tensor

df: The pandas dataframe to convert into a pytorch tensor
categorizer: The Categorizer to use while processing the categorical features in df

returns: df represented as a 2D pytorch tensor
"""
def transfrom_unprocessed_df_to_pytorch(df, categorizer, NUMERIC_FEATURES=None, CATEGORICAL_FEATURES=None):
  processed_df = transform_unprocessed_df_to_processed_df(df, categorizer, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
  data_tensor = transform_processed_df_to_pytorch(processed_df)
  return data_tensor
"""
Takes an file path and turns it into a pytorch tensor

file_path: A path to a csv file
categorizer: The Categorizer to use while processing the categorical features in the csv file pointed to by file_path

returns: df represented as a 2D pytorch tensor
"""
def transform_file_to_pytorch(unprocessed_df, categorizer, NUMERIC_FEATURES=None, CATEGORICAL_FEATURES=None):
  # unprocessed_df = transform_file_to_unprocessed_df(file_path)
  processed_df = transform_unprocessed_df_to_processed_df(unprocessed_df, categorizer, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
  data_tensor = transform_processed_df_to_pytorch(processed_df)
  return data_tensor


def new_filter_data(df, num_conds=dict(), cat_conds=dict(), text_conds=dict()):
    
    """
    df: DataFrame to filter
    num_conds: Numerical Conditions that we get from webpage
    cat_conds: Categorical Conditions that we get from webpage
    
    Returns the filtered DataFrame
    """
    
    
    for i in list(cat_conds.keys()):
        unique_cats = list(df[i].unique())
        df = df[df[i].isin(cat_conds[i])]
    
    for i in list(num_conds.keys()):
        if '-' in num_conds[i]:
            splits = num_conds[i].split('-')
            limits = [df[i].min() if splits[0] == '' else float(splits[0]), df[i].max() if splits[1] == '' else float(splits[1])]
            
            df = df[(df[i]>=limits[0]) & (df[i]<=limits[1])]
        else:
            try:
                df = df[df[i] == float(num_conds[i])]
            except:
                print("Invalid format to filter Numerical Columns")
                return None

    for i in list(text_conds.keys()):
        # try:
        df = df[df[i].apply(lambda x: all(word in x for word in text_conds[i]))]
        # df = df[df[i].str.contains(text_conds[i],case=False,na=False)]
        # except:
        #     print("Invalid format to filter Text Columns")
        #     return None
    
    return df




# def filter_desc(dict_name,df):
#     ft_values,op_list,cond_list = get_lists(dict_name,df)
#     filters = []
#     for i in range(len(ft_values)):
#         if op_list[i] == "range":
#             filters.append("Range of values between "+str(cond_list[i])+" on "+ft_values[i])
#         else:
#             filters.append(ft_values[i]+" is "+cond_list[i])
#     return filters


# def filter_data(features_list,operator_list,conditions_list,df):
#     ''' 
#     Inputs: 

#     Features_list: list of features or column names as string.
#     Ex: ['Age','Cholesterol','Sex','RestingECG']

#     Operator_list: list of operators as string.
#     Ex: ['<','range','=','=']

#     Conditions_list: list of Conditions on features. For numerical features the format is (min_val,max_val). For categorical features the format is one of the categories as string.
#     Ex: [60,(100,200),'M','Normal']

#     Df: whole population dataset taken as pandas data frame object.

#     Output: returns the filtered dataset as a pandas data frame object.

#     '''
#     for i in range(len(features_list)):
#         if operator_list[i] == '=':
#             df = df[df[features_list[i]] == conditions_list[i]]
#         elif operator_list[i] == 'range':
#             df = df[(df[features_list[i]] >= conditions_list[i][0]) & 
#                                                (df[features_list[i]] <= conditions_list[i][1])]
#     df = df.reset_index(drop=True)
    
#     return df


# def get_lists(dict_name,df):
#     """
#     dict_name: Dictionary from front-end or user interface with key value as feature name (String) and value as conditional value (String) to be applied on the column.
#     Ex: {
#         "Age":"-",
#         "Sex":"M"
#     }

#     feature_values: Dictionary for numerical features in the data with key value as feature name (String) and value as list of min and max value possible for that feature.
#     Ex: {
#         "Age" : [28,67],
#         "Cholesterol": [0.0,564.0]
#     }
#     """
    
#     features_list = list(dict_name.keys())
#     values = list(dict_name.values())
#     operators_list = []
#     conditions_list = []
#     for i in range(len(values)):
#         if '-' in values[i]:
#             operators_list.append('range')
#             val1,val2 = values[i].split('-',1)
#             if val1=='' or val2=='':
#                 _,numerical_ft = determine_dtype_ft(df)
#                 feature_values = dict()
#                 for i in numerical_ft:
#                     feature_values[i] = [df[i].min(),df[i].max()]
#         # print(val1,val2)
#             if val1 == '':
#                 val1 = feature_values[features_list[i]][0]
#                 val2 = eval(val2)
#             if val2 == '':
#                 val2 = feature_values[features_list[i]][1]
#                 val1 = eval(val1)
#             # print(val1,val2)
#             conditions_list.append((float(val1),float(val2)))
#         else:
#             operators_list.append('=')
#             try:
#                 conditions_list.append(float(values[i]))
#                 # print(eval(values[i]))
#             except ValueError:
#                 # print(values[i])
#                 conditions_list.append(values[i])
#     return features_list,operators_list,conditions_list

def read_data(model_name, production_run):
    
    data_csv = f"./pages/models/{model_name}/Ground Truths/{production_run}.csv"
    production_csv = f"./pages/models/{model_name}/Production Runs/{production_run}.csv"
    baseline = f"./pages/models/{model_name}/baseline.csv"
    
    actual_data = pd.read_csv(data_csv)
    production_data = pd.read_csv(production_csv)
    baseline_data = pd.read_csv(baseline)
    if 'Unnamed: 0' in actual_data.columns:
        actual_data = actual_data.drop('Unnamed: 0', axis=1)
    if 'Unnamed: 0' in production_data.columns:
        production_data = production_data.drop('Unnamed: 0', axis=1)
    if 'Unnamed: 0' in baseline_data.columns:
        baseline_data = baseline_data.drop('Unnamed: 0', axis=1)  
    return actual_data, production_data, baseline_data


def get_labels(baseline_data, model_name):
    if model_name == "Heart Disease Prediction":
        return ['Disease present', 'Disease not present']
    else:
        return list(baseline_data['target'].unique())

def EncodeLabels(gt_label, pd_label):
    
    label = LabelEncoder()
    gt = label.fit_transform(gt_label)
    pd = label.transform(pd_label)
    return gt, pd

def one_hot(data):
    if 'object' in [data[i].dtype for i in data]:
        object_data = data.select_dtypes(include='object')
        object_df = pd.get_dummies(object_data, dtype=int)
        rest_data = data.select_dtypes(exclude='object')
        return pd.concat([object_df, rest_data],axis=1)
    else:
        return data

def regression_metrics(true_labels,predicted_labels):
    metrics = dict()
    metrics['Mean Absolute Error'] = int(round(mean_absolute_error(true_labels, predicted_labels),2))
    # metrics['Mean Squared Error'] = np.round(mean_squared_error(true_labels, predicted_labels), decimals=2)
    metrics['Root Mean Squared Error'] = int(np.round(np.sqrt(float(mean_squared_error(true_labels, predicted_labels))), decimals=2))
    metrics['R2 Score'] = np.round(r2_score(true_labels, predicted_labels), decimals=2)
    metrics['Mean Percentage Error'] = np.round(np.mean((true_labels - predicted_labels) / true_labels) * 100, decimals=2)
    metrics['Mean Absolute Percentage Error'] = np.round(np.mean(np.abs((true_labels - predicted_labels) / true_labels)) * 100, decimals=2)
    return metrics

def classification_metrics(true_labels, predicted_labels, labels=None, pos_label=1):
    metrics = dict()
    metrics['report'] = classification_report(true_labels, predicted_labels, output_dict=True)
    if len(set(true_labels)) > 2 or len(set(true_labels)) > 2: 
        lb = LabelBinarizer()
        true_m = lb.fit_transform(true_labels)
        pred_m = lb.transform(predicted_labels)
        
        metrics['Accuracy'] = np.round(accuracy_score(true_labels, predicted_labels)*100, decimals=2)
        metrics['Precision'] = np.round(precision_score(true_m, pred_m, average="weighted")*100, decimals=2)
        metrics['Recall'] = np.round(recall_score(true_m, pred_m, average="weighted")*100, decimals=2)
        metrics['F1'] = np.round(f1_score(true_m, pred_m, average="weighted")*100, decimals=2)
        
        try:
            metrics['ROC_AUC'] = np.round(roc_auc_score(true_m, pred_m, multi_class="ovr")*100, decimals=2)
        except:
            metrics['ROC_AUC'] = 0
        
        cm = confusion_matrix(true_labels, predicted_labels)
        FP = np.sum(cm, axis=0) - np.diag(cm)
        TN = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + np.diag(cm)

        total_FP = np.sum(FP)
        total_TN = np.sum(TN)
    
        metrics['False Positive Rate'] = np.round((total_FP / (total_FP + total_TN))*100, decimals=2)
        
        return metrics
    
    else:
        
        metrics['Accuracy'] = np.round(accuracy_score(true_labels, predicted_labels)*100, decimals=2)
        metrics['Precision'] = np.round(precision_score(true_labels, predicted_labels, pos_label=pos_label)*100, decimals=2)
        metrics['Recall'] = np.round(recall_score(true_labels, predicted_labels, pos_label=pos_label)*100, decimals=2)
        metrics['F1'] = np.round(f1_score(true_labels, predicted_labels, pos_label=pos_label)*100, decimals=2)
        try:
            
            metrics['ROC_AUC'] = np.round(roc_auc_score(true_labels, predicted_labels, pos_label=pos_label)*100, decimals=2)
        except:
            metrics['ROC_AUC'] = 0
        
        try:
            true_negative, false_positive, false_negative, true_positive = confusion_matrix(true_labels, predicted_labels).ravel()
            metrics['False Positive Rate'] = np.round((false_positive / (false_positive + true_negative))*100, decimals=2)
        except:
        # metrics['True Positive Rate'] = true_positive / (true_positive + false_negative)
            metrics['False Positive Rate'] = 0
        
        return metrics

# def plot_reg_target(true_labels, predicted_labels):
#     x = np.arange(len(true_labels)) 
#     figure = go.Figure()
#     figure.add_trace(go.Scatter(x=x, y=true_labels,
#                     mode='markers',
#                     name='Ground Truth'))
#     figure.add_trace(go.Scatter(x=x, y=predicted_labels,
#                     mode='markers',
#                     name='Predicted'))
#     return figure

def plot_reg_target(true_labels, predicted_labels):
    x = np.arange(len(true_labels))
    figure = go.Figure()

    for i in range(len(x)):
        figure.add_trace(go.Scatter(
            x=[x[i], x[i]],
            y=[true_labels[i], predicted_labels[i]],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))

    figure.add_trace(go.Scatter(
        x=x, y=true_labels,
        mode='markers',
        name='Ground Truth',
        marker=dict(color='blue')
    ))

    figure.add_trace(go.Scatter(
        x=x, y=predicted_labels,
        mode='markers',
        name='Predicted',
        marker=dict(color='red')
    ))

    figure.update_layout(
        title='Ground Truth vs Production with Connections',
        xaxis_title='Instance',
        yaxis_title='Value',
        template='plotly_white'
    )

    return figure

def plot_bar(production,benchmark):

    figure = go.Figure()
    figure.add_trace(go.Bar(name='Production Run', x=list(production.keys()), y=list(production.values()), text=list(production.values())))
    figure.add_trace(go.Bar(name='Benchmark', x=list(benchmark.keys()), y=list(benchmark.values()), text=list(benchmark.values())))
    return figure

def plot_confusion_matrix(true_labels,predicted_labels, class_labels):
    
    cm_df = pd.DataFrame(confusion_matrix(true_labels, predicted_labels), index=class_labels, columns=class_labels)
    fig = px.imshow(cm_df,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=class_labels,
                    y=class_labels,
                    text_auto=True)
    return fig

def plot_roc(true_labels, predicted_labels, pos_label=1):
    
    if len(set(true_labels)) > 2 or len(set(true_labels)) > 2: 
        lb = LabelBinarizer()
        true_m = lb.fit_transform(true_labels)
        pred_m = lb.transform(predicted_labels)
        
        
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(len(lb.classes_)):
            fpr[i], tpr[i], _ = roc_curve(true_m[:, i], pred_m[:, i])
            roc_auc[lb.classes_[i]] = round(auc(fpr[i], tpr[i]), 2)

        figure = px.area(x=fpr, y=tpr,
            title=f'ROC Curve (AUC={roc_auc})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500)
        figure.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)
        return figure, roc_auc
    
    else:
        
        false_positve_r, true_positive_r, thresholds = roc_curve(true_labels, predicted_labels, pos_label=pos_label)
        roc_auc = auc(false_positve_r, true_positive_r)
        figure = px.area(x=false_positve_r, y=true_positive_r,
            title=f'ROC Curve (AUC={round(roc_auc, 2)})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500)
        figure.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)
        return figure, roc_auc

def plot_PR(true_labels, predicted_labels):
    
    if len(set(true_labels)) > 2 or len(set(true_labels)) > 2 : 
        lb = LabelBinarizer()
        true_m = lb.fit_transform(true_labels)
        pred_m = lb.transform(predicted_labels)
    
        precision, recall, average_precision = dict(), dict(), dict()
        
        for i in range(len(lb.classes_)):
            precision[i], recall[i], _ = precision_recall_curve(true_m[:, i], pred_m[:, i])
            average_precision[lb.classes_[i]] = round(average_precision_score(true_m[:, i], pred_m[:, i]), 2)
        
        average = {k:round(i*100,2) for k,i in average_precision.items()}
        
        figure = px.area(x=[i[0] for i in list(recall.values())], y=[i[0] for i in list(precision.values())],
            title=f'Precision: {average}',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500)
        figure.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0)
        
        return figure, precision, recall

    else:
        
        precision, recall, _ = precision_recall_curve(true_labels, predicted_labels)
        precision_ = np.round(precision_score(true_labels, predicted_labels)*100, decimals=2)
        recall_ = np.round(recall_score(true_labels, predicted_labels)*100, decimals=2)
        figure = px.area(x=recall, y=precision,
            title=f'Precision: {precision}, Recall: {recall}',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500)
        figure.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0)
        return figure, precision_, recall_

def data_completeness(df):
    completeness = round((len(df.dropna()) / len(df)) * 100, 2)
    missing_data = df.isnull().sum()
    return completeness, len(df), len(df.dropna()), len(df) - len(df.dropna()), missing_data

def donut_for_completeness(value):
    df = {"Label": ["Complete", "Missing"], "Value": [value, 100 - value]}
    df = pd.DataFrame(df)
    figure = px.pie(df, names='Label', values='Value', hole=0.8)
    return figure

def plot_missing_bar(series, len_df):
    df = {"Feature": [i for i in series.index], "Values": [round((i / len_df), 4) * 100  for i in series]}
    df = pd.DataFrame(df)
    figure = px.bar(df, x="Feature", y="Values", text_auto=True)
    return figure

def data_uniqueness(df):
    score = round((len(df.drop_duplicates()) / len(df)) * 100, 2)
    uniqueness_score, uniqueness_value = dict(), dict()
    uniqueness_score["Feature"] = [i for i in df]
    uniqueness_score["Value"] = [round((df[i].nunique() / len(df[i])) * 100, 4) for i in df]
    uniqueness_value["Feature"] = [i for i in df]
    uniqueness_value["Value"] = [len(df[i].unique()) for i in df]
    return score, uniqueness_score, uniqueness_value, len(df), len(df.drop_duplicates())

def donut_for_uniqueness(value):
    df = {"Label": ["Unique", "Duplicate"], "Value": [value, 100 - value]}
    df = pd.DataFrame(df)
    figure = px.pie(df, names='Label', values='Value', hole=0.8)
    return figure

def donut_for_validity(value):
    df = {"Label": ["Valid Data", "Invalid Data"], "Value": [value, 100 - value]}
    df = pd.DataFrame(df)
    figure = px.pie(df, names='Label', values='Value', hole=0.8)
    return figure

def plot_unique_bar(uni_dict):
    figure = px.bar(x=uni_dict["Feature"], y=uni_dict["Value"], text_auto=True)
    return figure

def outliers_for_num_data(data):
    
    outliers_indices = {i:[] for i in data.columns}
    
    for i in data.select_dtypes(exclude=['object', 'bool']).columns:
        
        Q1 = data[i].quantile(0.25)
        Q3 = data[i].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[i] < lower_bound) | (data[i] > upper_bound)]
        outliers_indices[i] = outliers.index.tolist()
    
    return outliers_indices
    
def validity_check(baseline_data, production_data):
    
    baseline_data = baseline_data.loc[:, ~baseline_data.columns.str.contains('^Unnamed')]
    common_columns = baseline_data.columns.intersection(production_data.columns)
    production_data = production_data[common_columns]
    
    miss_dict = {i:[] for i in baseline_data.columns}
    cat_has_int = {i:[] for i in baseline_data.select_dtypes(include='object').columns}
    int_has_cat = {i:[] for i in baseline_data.select_dtypes(exclude='object').columns}
    indices = []
    outliers_index = {i:[] for i in baseline_data.columns}
    
    if list(baseline_data.columns) == list(production_data.columns):
        for column in baseline_data.columns:
            
            if production_data[column].isnull().any():
                missing_indices = production_data[production_data[column].isnull()].index.tolist()
                
                miss_dict[column] = missing_indices
            
            if production_data[column].dtype == 'object' and baseline_data[column].dtype == 'object':
                unique_bs = set(baseline_data[column].unique())
                unique_pd = set(production_data[column].unique())
                extras = unique_bs.symmetric_difference(unique_pd)
                
                cat_has_int_id = []
                for i in extras:
                    index_list = production_data[production_data[column]  ==  i].index.tolist()
                    cat_has_int_id.extend(index_list)
                cat_has_int[column] = cat_has_int_id
            
            # if baseline_data[column].dtype != production_data[column].dtype:
            #     int_has_cat_id = []
            #     for index, value in production_data[column].items():
                    
            #         try:
            #             float(value)
            #         except ValueError:
            #             int_has_cat_id.append(index)
            #     int_has_cat[column] = int_has_cat_id
                        
    for k in list(miss_dict.keys()):
        indices.extend(miss_dict[k])
    
    for k in list(cat_has_int.keys()):
        indices.extend(cat_has_int[k])
    
    # for k in list(int_has_cat.keys()):
    #     indices.extend(int_has_cat[k])
    
    num_invalid = len(set(indices))
    total_rows = len(production_data)
    score = np.round((total_rows - num_invalid) / total_rows * 100, 2)
    
    if num_invalid != 0:
        clean_df = production_data.drop(indices, axis=0)
        outliers_index = outliers_for_num_data(clean_df)
    
    return cat_has_int, miss_dict, score, indices, outliers_index, num_invalid

def plot_validity_bar(f_name, missing, wrong, BR):
    figure = go.Figure()
    figure.add_trace(go.Bar(name="Missing", x=[f_name], y=[missing], text=[missing]))
    figure.add_trace(go.Bar(name="Wrong Data Types", x=[f_name], y=[wrong], text=[wrong]))
    figure.add_trace(go.Bar(name="Outliers", x=[f_name], y=[BR], text=[BR])) 
    return figure

def ks_test(baseline, production, num_ft=None, nlp=False, alpha=0.05, b_name="Baseline Data"):
    
    if nlp == True:
        x = "Frequency"
    else:
        x = "Target Values"
    if num_ft == None or nlp == True:
        ks_stat, p_value = ks_2samp(baseline, production)
        results = {'ks-stat': round(ks_stat,2), 'ks_p-value': round(p_value,2), 'ks_status': bool(p_value < alpha)}
        
        sample1 = np.sort(baseline)
        sample2 = np.sort(production)
        
        cdf1 = np.arange(1, len(sample1) + 1) / len(sample1)
        cdf2 = np.arange(1, len(sample2) + 1) / len(sample2)
        
        figure = go.Figure() 
        figure.add_trace(go.Scatter(x=sample1, y=cdf1, mode='lines', name=b_name))
        figure.add_trace(go.Scatter(x=sample2, y=cdf2, mode='lines', name="Production Data"))
        figure.update_layout(
                        xaxis_title=x,
                        yaxis_title='Cumulative Distribution ',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=500)
        figure.update_xaxes(tickfont={"size":14, "color":"black"})
        figure.update_yaxes(tickfont={"size":14, "color":"black"})
        
        return results, figure
    else:
        results = dict()
        figures = dict()
        for i in num_ft:
            ks_stat, p_value = ks_2samp(baseline[i], production[i])
            results[i] = {'ks-stat': round(ks_stat,2), 'ks_p-value': round(p_value,2), 'ks_status': bool(p_value < alpha)}
            
            sample1 = np.sort(baseline[i])
            sample2 = np.sort(production[i])
            
            cdf1 = np.arange(1, len(sample1) + 1) / len(sample1)
            cdf2 = np.arange(1, len(sample2) + 1) / len(sample2)
            
            figure = go.Figure() 
            figure.add_trace(go.Scatter(x=sample1, y=cdf1, mode='lines', name=b_name))
            figure.add_trace(go.Scatter(x=sample2, y=cdf2, mode='lines', name="Production Data"))
            figure.update_layout(
                            xaxis_title='Feature Values',
                            yaxis_title='Cumulative Distribution ',
                            # title_font={"size": 20},
                            xaxis_title_font={"size":16, "color":"black"},
                            yaxis_title_font={"size":16, "color":"black"},
                            width=1080,
                            height=500)
            figure.update_xaxes(tickfont={"size":14, "color":"black"})
            figure.update_yaxes(tickfont={"size":14, "color":"black"})
            
            figures[i] = figure                 
        return results, figures

def chi_test(baseline, production, categorical_ft=None, alpha=0.05):

    if categorical_ft == None:
        cross_tab = pd.crosstab(baseline, production)
        
        chi_stat, p_value, _, _ = chi2_contingency(cross_tab)
        results = {'chi': round(chi_stat, 2), 'chi_p-value': round(p_value,2), 'chi_status': bool(p_value > alpha)}
        
    
        baseline_dict = dict(baseline.value_counts())
        prod_dict = dict(production.value_counts())


        baseline_unique, prod_unique = list(baseline_dict.keys()), list(prod_dict.keys())
        features = list(set(baseline_unique + prod_unique))
    
        baseline_props = {j:baseline_dict[j] if j in baseline_unique else 0 for j in features} 
        prod_props = {j:prod_dict[j] if j in prod_unique else 0 for j in features}

        figure = go.Figure()
        figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
        figure.add_traces(go.Bar(x=list(prod_props.keys()), y=list(prod_props.values()), text=list(prod_props.values()), name="Production"))

        figure.update_layout(
                        xaxis_title=f'Categories in Target',
                        yaxis_title='Category Value Counts ',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=500)
        figure.update_xaxes(tickfont={"size":14, "color":"black"})
        figure.update_yaxes(tickfont={"size":14, "color":"black"})
        figure.update_layout(barmode='group')

        return results, figure
    
    else:
        results, figures = dict(), dict()
        
        for i in categorical_ft:
            
            cross_tab = pd.crosstab(baseline[i], production[i])
            
            chi_stat, p_value, _, _ = chi2_contingency(cross_tab)
            results[i] = {'chi': round(chi_stat, 2), 'chi_p-value': round(p_value,2), 'chi_status': bool(p_value > alpha)}

            baseline_dict = dict(baseline[i].value_counts())
            prod_dict = dict(production[i].value_counts())


            baseline_unique, prod_unique = list(baseline_dict.keys()), list(prod_dict.keys())
            features = list(set(baseline_unique + prod_unique))

            baseline_props = {j:baseline_dict[j] if j in baseline_unique else 0 for j in features} 
            prod_props = {j:prod_dict[j] if j in prod_unique else 0 for j in features}

            figure = go.Figure()
            figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
            figure.add_traces(go.Bar(x=list(prod_props.keys()), y=list(prod_props.values()), text=list(prod_props.values()), name="Production"))

            figure.update_layout(
                            xaxis_title=f'Categories in {i}',
                            yaxis_title='Category Value Counts ',
                            # title_font={"size": 20},
                            xaxis_title_font={"size":16, "color":"black"},
                            yaxis_title_font={"size":16, "color":"black"},
                            width=1080,
                            height=500)
            figure.update_xaxes(tickfont={"size":14, "color":"black"})
            figure.update_yaxes(tickfont={"size":14, "color":"black"})
            figure.update_layout(barmode='group')

            figures[i] = figure

        return results, figures


def determine_dtype_ft(df):
    categorical_ft, numerical_ft = [], []
    for i in df.columns:
        if df[i].dtype == 'object':
            categorical_ft.append(i)
        else:
            numerical_ft.append(i)
    return categorical_ft, numerical_ft


def pred_drift_plots(production, model_type):

    if model_type == "Regression":
        y = 'Statistic Value'
    if model_type == "Classification":
        y = 'Probability Value'
    
    t_max, t_min, t_mean, t_median = dict(), dict(), dict(), dict()
    for i in production:
        t_max[i] = round(max(production[i]),2)
        t_min[i] = round(min(production[i]),2)
        t_mean[i] = round(np.mean(production[i]),2)
        t_median[i] = round(np.median(production[i]),2)
        
    stats = {'Max': t_max, 'Min': t_min, 'Mean': t_mean, 'Median': t_median}

    figures = dict()
    for i in stats:
        figure = go.Figure()
        figure.add_traces(go.Bar(x=list(stats[i].keys()), y=list(stats[i].values()), text=list(stats[i].values())))
        figure.update_layout(title=f'{i} Statistic',
                        xaxis_title=f'Production Runs',
                        yaxis_title=f'{y}',
                        title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=400,
                        height=400)
        figure.update_xaxes(tickfont={"size":14, "color":"black"})
        figure.update_yaxes(tickfont={"size":14, "color":"black"})
        figure.update_layout(barmode='group')
        figures[i] = figure
    return stats, figures

def text_cleaning_with_numbers_as_text(text):
    lemmatize = WordNetLemmatizer()
    text = text.lower()
    contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would",
                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    text = ' '.join([contractions[word] if word in contractions else word for word in text.split()])
    text = re.sub(r'[^0-9a-zA-Z\s]+','',text)
    text = text.translate(str.maketrans('','',string.punctuation))
    text = ' '.join([num2words(word) if word.isdigit() else word for word in text.split()])
    while re.search('-',text):
        text = re.sub('-',' ',text)
    words = nltk.word_tokenize(text)
    stopword = set(stopwords.words('english'))
    new_words = [word for word in words if word not in stopword]
    lem_words = [lemmatize.lemmatize(word) for word in new_words]
    return ' '.join(lem_words)

def load_model(model_name):
    if model_name == "Medical Cost Prediction":
        return joblib.load(f"C:/Users/Akshat Mittu/Desktop/Model Monitoring Dashboard/pages/models/{model_name}/model.joblib")

def syntax_drift(results_b, results_pr):
    
    baseline_freq = Counter(results_b)
    production_freq = Counter(results_pr)
    
    baseline_cloud = WordCloud(width=650, height=350, background_color="white").generate(' '.join([i for i in results_b]))
    prod_cloud = WordCloud(width=650, height=350, background_color="white").generate(' '.join([i for i in results_pr]))
    
    top_5_baseline = dict(sorted(baseline_freq.items(), key=lambda a:a[1], reverse=True)[:5])
    top_5_production = dict(sorted(production_freq.items(), key=lambda a:a[1], reverse=True)[:5])
    
    uncommon = set(results_pr) - set(results_b)
    
    uncommon_freq = {i: production_freq[i] for i in uncommon}
    top_5_uncommon = dict(sorted(uncommon_freq.items(), key=lambda a:a[1], reverse=True)[:5])
    
    figure_bvp = go.Figure()
    figure_bvp.add_trace(go.Bar(name='Baseline Words', x=list(top_5_baseline.keys()), y=list(top_5_baseline.values()), text=list(top_5_baseline.values())))
    figure_bvp.add_trace(go.Bar(name='Production Words', x=list(top_5_baseline.keys()), y=[production_freq[i] for i in top_5_baseline], text=[production_freq[i] for i in top_5_baseline]))
    figure_bvp.update_layout(
                        #title=f'Frequencies of Top 5 Baseline words in Production',
                        xaxis_title='Words in Baseline Data',
                        yaxis_title='Count',
                        #title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=450)
    figure_bvp.update_xaxes(tickfont={"size":14, "color":"black"})
    figure_bvp.update_yaxes(tickfont={"size":14, "color":"black"})
    figure_bvp.update_layout(barmode='group')
                         
    figure_pvb = go.Figure()
    figure_pvb.add_trace(go.Bar(name='Production Words', x=list(top_5_production.keys()), y=list(top_5_production.values()), text=list(top_5_production.values())))
    figure_pvb.add_trace(go.Bar(name='Baseline Words', x=list(top_5_production.keys()), y=[baseline_freq[i] for i in top_5_production], text=[baseline_freq[i] for i in top_5_production]))
    figure_pvb.update_layout(
                        #title=f'Frequencies of Top 5 Production words in Baseline',
                        xaxis_title='Words in Production Data',
                        yaxis_title='Count',
                        #title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=450)
    figure_pvb.update_xaxes(tickfont={"size":14, "color":"black"})
    figure_pvb.update_yaxes(tickfont={"size":14, "color":"black"})
    figure_pvb.update_layout(barmode='group')
                         
    figures = dict()
    figures['baseline cloud'] = baseline_cloud
    figures['production cloud'] = prod_cloud
    figures['top 5 baseline bar'] = figure_bvp
    figures['top 5 production bar'] = figure_pvb
    
    return figures, baseline_freq, production_freq, uncommon, uncommon_freq, top_5_uncommon, top_5_baseline, top_5_production

def preprocess_for_embedding(text):
    
    text = text.lower()
    contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would",
                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    text = ' '.join([contractions[word] if word in contractions else word for word in text.split()])
    text = re.sub(r'[^0-9a-zA-Z\s]+','',text)
    text = text.translate(str.maketrans('','',string.punctuation))
    text = ' '.join([num2words(word) if word.isdigit() else word for word in text.split()])
    while re.search('-',text):
        text = re.sub('-',' ',text)
    while re.search(',',text):
        text = re.sub(',',' ',text)
    return text

def creat_embed(text, use):
    embeds = use([text])
    return embeds[0].numpy()

def create_and_save_embeddings(text_df, text_col, file_name):
    
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use = hub.load(module_url)
    
    text_df['embeds'] = text_df[text_col].apply(preprocess_for_embedding) 
    
    embed_df = dict()
    for i in range(len(text_df)):
        embed_df[i] = creat_embed(text_df.iloc[i]['embeds'], use).tolist()
    
    df = pd.DataFrame(embed_df).T
    
    df.to_csv(f"{file_name}.csv", index=False)


def semantic_drift(production_run, model_name, col='text', b_name="Baseline"):
    
    if col == 'text':
        baseline = pd.read_csv(f"./pages/models/{model_name}/embeds/baseline.csv")
        production = pd.read_csv(f"./pages/models/{model_name}/embeds/{production_run}.csv")
    elif col == 'target':
        baseline = pd.read_csv(f"./pages/models/{model_name}/embeds_target/Ground Truths/{production_run}.csv")
        production = pd.read_csv(f"./pages/models/{model_name}/embeds_target/Production Runs/{production_run}.csv")
        
    if 'Unnamed: 0' in production.columns:
        production.drop('Unnamed: 0', axis=1, inplace=True)
    if 'Unnamed: 0' in baseline.columns:
        baseline.drop('Unnamed: 0', axis=1, inplace=True)
    
    tsne_b = TSNE(n_components = 2, random_state=42)
    tsne_base = tsne_b.fit_transform(baseline[:1000])
    
    tsne_p = TSNE(n_components = 2, random_state=42)
    tsne_pr = tsne_p.fit_transform(production[:1000])
    
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[i[0] for i in tsne_base], y=[i[1] for i in tsne_base],
                    mode='markers',
                    name=b_name))
    figure.add_trace(go.Scatter(x=[i[0] for i in tsne_pr], y=[i[1] for i in tsne_pr],
                    mode='markers',
                    name='Production'))
    figure.update_layout(
                        #xaxis_title=x,
                        #yaxis_title=' ',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=500)
    
    if col == "target":
        return figure
    else:
        edges = []
        pr_node_sizes = dict()
        bs_nodes_used = []
        edge_alpha = dict()
        
        random_state = random.randint(1,100)
        
        baseline_sample = baseline.sample(15, random_state=random_state)
        prod_sample = production.sample(15, random_state=random_state)
        
        for i in range(len(prod_sample)):
            sim = cosine_similarity(X=[prod_sample.iloc[i]], Y=baseline_sample)
            indices = np.where(sim>0.45)[1]
            bs_nodes_used.extend(list(baseline_sample.iloc[indices].index))
            pr_node_sizes[prod_sample.index[i]] = len(indices) + 100
            for j in indices:
                edges.append((prod_sample.index[i], baseline_sample.index[j]))
                edge_alpha[(prod_sample.index[i], baseline_sample.index[j])] = sim[0][j]
            
        bs_node_sizes = {i:bs_nodes_used.count(i) + 100 for i in baseline_sample.index}
        
        isolated = set(prod_sample.index) - set([i for (i,j) in edges]) 
        isolated_score = round(len(isolated) / len(prod_sample), 2) * 100
        
        return figure, isolated_score, isolated, edges, edge_alpha, pr_node_sizes, bs_node_sizes
        
def text_cleaning(text):

    text = text.lower()
    contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would",
                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    text = ' '.join([contractions[word] if word in contractions else word for word in text.split()])
    text = re.sub(r'[^0-9a-zA-Z\s]+','',text)
    return text

def get_words(data, col='text'):
    
    cleaned = data[col].apply(text_cleaning)
    words = [i.split() for i in cleaned]
    results = []
    for i in words:
        results.extend(i) 

    return results

def check_spelling(words):
    
    checker = SpellChecker()
    misspelled = checker.unknown(words)
    
    index = random.randint(0,len(misspelled))
    
    return len(misspelled), round(len(set(misspelled)) / len(set(words)), 2) * 100, list(misspelled)[index:index+10]

def nlp_quality_metrics(helper, quality_type, col='text'):
    
    max_index, min_index = helper[quality_type].argmax(), helper[quality_type].argmin()
    max_text, min_text = helper[col].iloc[max_index], helper[col].iloc[min_index]
    
    ma, mi, avg = helper[quality_type].max(), helper[quality_type].min(), round(helper[quality_type].mean(), 2)
    
    return ma, mi, avg, max_text, min_text, max_index, min_index

def get_helper_csv_nlp(data, column='text', text_path=None, target_path=None):
    
    helper = dict()
    
    helper[column] = data[column]
    # helper['cleaned_text'] = data['text'].apply(text_cleaning_with_numbers_as_text)
    helper['length'] = [len(i.split()) for i in data[column]]
    helper['readability'] = [textstat.flesch_kincaid_grade(i) for i in data[column]]
    
    if (column == "target") and (text_path != None):
        sim_scores = []
        gt_text = pd.read_csv(text_path)
        gt_target = pd.read_csv(target_path)
        for i in range(len(gt_text)):
            sim = cosine_similarity(X=[gt_text.iloc[i]], Y=[gt_target.iloc[i]])
            sim_scores.append(round(sim[0][0], 2))
        
        helper['similarity'] = sim_scores
    
    # else:
    #     pass
        
    helper_df = pd.DataFrame(helper)
    return helper_df

def donut_for_spell_errors(value):
    df = {"Label": ["Spelling Error", "Correct"], "Value": [value, 100 - value]}
    df = pd.DataFrame(df)
    figure = px.pie(df, names='Label', values='Value', hole=0.8)
    return figure

def get_glcm_csv(path_df, distances, angles):
    
    """
    Takes in image directory paths as df then applies one distance to one angle not a list
    """
    contrast_list, energy_list, homogeneity_list, correlation_list, dissimilarity_list = [], [], [], [], []
    
    for i in path_df['paths']:
        image = cv.imread(i)
    
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, prop="contrast")
        energy = graycoprops(glcm, prop='energy')
        homogeneity = graycoprops(glcm, prop='homogeneity')
        correlation = graycoprops(glcm, prop='correlation')
        dissimilarity = graycoprops(glcm, prop='dissimilarity')
        
        contrast_list.append(contrast[0][0])
        energy_list.append(energy[0][0])
        homogeneity_list.append(homogeneity[0][0])
        correlation_list.append(correlation[0][0])
        dissimilarity_list.append(dissimilarity[0][0])
        
    path_df['contrast'] = contrast_list
    path_df['energy'] = energy_list
    path_df['homogeneity'] = homogeneity_list
    path_df['correlation'] = correlation_list
    path_df['dissimilarity'] = dissimilarity_list
    
    return path_df

def prep_and_predict(model,df,class_names,image_size=224):
    
    prediction = []
    probs = []
    for i in df['paths']:
        image = tf.io.read_file(i) # Reading the file
        image = tf.image.decode_image(image,channels=3) # Decode the read image to tensor and making sure to have 3 channels due to rgb
        image = tf.image.resize(image,size=(image_size,image_size))

        image = image/255. # Rescaling the data
        
        proba = model.predict(tf.expand_dims(image,axis=0))
        probs.append(round(proba[0][0],2)) # Expanding the dimensions to make sure it matches with the dimensions of the input data
        
        
        pred_class = class_names[int(tf.round(proba)[0][0])]
        
        prediction.append(pred_class)
    
    df['probs'] = probs
    df['target'] = prediction
    
    return df

def CV_data_quality(paths_df):
    
    resolutions, sharpness, brightness, size, noise, num_anomaly, is_anomaly = [], [], [], [] ,[], [], []
    
    for i in paths_df['paths']:
        
        image = cv.imread(i)
        
        resolution = (image.shape[1], image.shape[0])
        size_ = image.size
        
        sharp = cv.Laplacian(image, cv.CV_64F).var()
        bright = np.mean(image)
        noise_ = np.std(image)
        
        deviation = np.abs(image - np.mean(image)) / np.std(image)
        
        num_anomaly_ = np.sum(deviation > 3)
        is_anomaly_= num_anomaly_ > 2500
        
        resolutions.append(resolution)
        sharpness.append(sharp)
        brightness.append(bright)
        size.append(size_)
        noise.append(noise_)
        num_anomaly.append(num_anomaly_)
        is_anomaly.append(is_anomaly_)
        
    paths_df['resolution'] = resolutions
    paths_df['sharpness'] = sharpness
    paths_df['brightness'] = brightness
    paths_df['size'] = size
    paths_df['noise'] = noise
    paths_df['Number of Anomalies'] = num_anomaly
    paths_df['Anomaly'] = is_anomaly

    return paths_df

def CV_images_for_data_quality(paths_df):
    
    normal_path = paths_df[paths_df['Anomaly'] == False].sample(1)['paths'].iloc[0]
    anomaly_path = paths_df[paths_df['Anomaly'] == True].sample(1)['paths'].iloc[0]
    
    figures = dict()
    
    figures['max sharpness'] = cv.imread(paths_df.iloc[paths_df['sharpness'].argmax()]['paths'])
    figures['min sharpness'] = cv.imread(paths_df.iloc[paths_df['sharpness'].argmin()]['paths'])
    
    figures['max brightness'] = cv.imread(paths_df.iloc[paths_df['brightness'].argmax()]['paths'])
    figures['min brightness'] = cv.imread(paths_df.iloc[paths_df['brightness'].argmin()]['paths'])
    
    figures['max noise'] = cv.imread(paths_df.iloc[paths_df['noise'].argmax()]['paths'])
    figures['min noise'] = cv.imread(paths_df.iloc[paths_df['noise'].argmin()]['paths'])
    
    normal_image = cv.imread(normal_path)
    anomaly_image = cv.imread(anomaly_path)
    
    gray_normal = cv.cvtColor(normal_image, cv.COLOR_BGR2GRAY)
    pixels_normal = gray_normal.flatten()
    
    gray_anomaly = cv.cvtColor(anomaly_image, cv.COLOR_BGR2GRAY)
    pixels_anomaly = gray_anomaly.flatten()
    
    
    figure = go.Figure()
    figure.add_traces(go.Histogram(x=pixels_normal, nbinsx=256, name="Normal Image"))
    figure.add_traces(go.Histogram(x=pixels_anomaly, nbinsx=256, name="Anomaly Image"))
    
    figure.update_layout(barmode='overlay',
                        title="Histogram of Pixel Intensities",
                        xaxis_title="Pixel Intensity")
    
    figures['Histogram'] = figure
    
    return figures

def get_text_scores_df(gt, prod, gt_embed, prod_embed, gt_input):

  bleu_scores, rouge_r, rouge_p, rouge_f, sim_scores, sim_text_scores = [], [], [], [], [], []
  
  rouge = Rouge()

  for i in range(len(gt)):

    bleu_scores.append(round(sentence_bleu(gt['target'].iloc[i].split(), prod['target'].iloc[i].split()), 4))

    rouge_score = rouge.get_scores(gt['target'].iloc[i], prod['target'].iloc[i])[0]

    rouge_r.append(round(rouge_score['rouge-1']['r'], 2))
    rouge_p.append(round(rouge_score['rouge-1']['p'], 2))
    rouge_f.append(round(rouge_score['rouge-1']['f'], 2))

    sim = cosine_similarity(X=[gt_embed.iloc[i]], Y=[prod_embed.iloc[i]])
    sim_scores.append(round(sim[0][0], 2))
    
    another_sim = cosine_similarity(X=[gt_input.iloc[i]], Y=[prod_embed.iloc[i]])
    sim_text_scores.append(round(another_sim[0][0], 2))

  text_scores_df = pd.DataFrame({"bleu": bleu_scores,
                                "rouge_recall": rouge_r,
                                 "rouge_precision": rouge_p,
                                 "rouge_F1": rouge_f,
                                "similarity_with_gt": sim_scores,
                                "similarity_wth_text": sim_text_scores})
  return text_scores_df

def text_score_figures(scores_df):

  figures = dict()

  figure_hist_rouge = go.Figure()

  figure_hist_rouge.add_traces(go.Histogram(x=scores_df['rouge_recall'], nbinsx=10, name="Recall"))
  figure_hist_rouge.add_traces(go.Histogram(x=scores_df['rouge_precision'], nbinsx=10, name="Precision"))
  figure_hist_rouge.add_traces(go.Histogram(x=scores_df['rouge_F1'], nbinsx=10, name="F1"))

  figure_hist_rouge.update_layout(barmode='overlay',
                        title="ROUGE Scores",
                        xaxis_title="Score")

  figure_hist_bleu = go.Figure()
  figure_hist_bleu.add_traces(go.Histogram(x=scores_df['bleu'], nbinsx=10, name="BLEU Score"))

  figure_hist_similarity = go.Figure()
  figure_hist_similarity.add_traces(go.Histogram(x=scores_df['similarity_with_gt'], nbinsx=10, name="Similarity with Ground Truth"))
  
  figure_hist_similarity_t = go.Figure()
  figure_hist_similarity_t.add_traces(go.Histogram(x=scores_df['similarity_wth_text'], nbinsx=10, name="Similarity with Text"))

  figures['ROUGE Box'] = px.box(scores_df,['rouge_F1', 'rouge_recall', 'rouge_precision'], points="all")
  figures['ROUGE Hist'] = figure_hist_rouge

  figures['BLEU Box'] = px.box(scores_df, 'bleu', points="all")
  figures['BLEU Hist'] = figure_hist_bleu

  figures['Similarity Box'] = px.box(scores_df, 'similarity_with_gt', points="all")
  figures['Similarity Hist'] = figure_hist_similarity
  
  figures['Similarity_t Box'] = px.box(scores_df, 'similarity_wth_text', points="all")
  figures['Similarity_t Hist'] = figure_hist_similarity_t

  return figures

def seq2seq_metrics(gt_target, prod_target):

  metrics = dict()
  rouge = Rouge()
  scores = rouge.get_scores(gt_target, prod_target, avg=True)

  metrics['Precision'] = round(scores['rouge-1']['p'] * 100, 2)
  metrics['F1'] = round(scores['rouge-1']['f'] * 100, 2)
  metrics['Recall'] = round(scores['rouge-1']['r'] * 100, 2)

  metrics['BLEU'] = round(corpus_bleu(gt_target, prod_target), 2)

  return metrics

# Shifting Mean
def shift_mean(data, alpha=0):
  return data + (alpha * np.std(data))

# Shifting Variance
def shift_var(data, alpha=0):
  mean = np.mean(data)
  return mean + (data - mean) * (1 + alpha)

# Is this right? Or should we transform random points in the data (is this same as inducing outliers?)
def shift_skewness(data, alpha=0):
    return np.sign(data) * np.abs(data) ** (1 + alpha / 2)

# Add outliers
def induce_outliers(data, alpha=0):
  
    num_outliers = int(alpha * len(data)) 
    outlier_indices = np.random.choice(len(data), size=num_outliers, replace=False)

    outliers = np.mean(data) + random.randint(-5,5) * np.std(data) * np.random.randn(num_outliers)
    data[outlier_indices] = outliers

    return data

def determine_dtype_ft(df):
    categorical_ft, numerical_ft = [], []
    for i in df.columns:
        if df[i].dtype == 'object':
            categorical_ft.append(i)
        else:
            numerical_ft.append(i)
    return categorical_ft, numerical_ft

def categorical_shift(data, alpha=0, new_cat=False, max_new_cat=1):
  
  if new_cat:
    unique_cats = list(data.unique()) + [f'NewCat{i}' for i in range(max_new_cat)]
  else:
    unique_cats = list(data.unique())
    
  indices = np.random.choice(len(data), size=int(alpha * len(data)), replace=False)

  for i in indices:
    current_value = data.iloc[i]
    new_value = np.random.choice(list(set(unique_cats) - set(current_value)))
    data.iloc[i] = new_value

  return data

def induce_drift(data, features=None, alpha_mean=0, alpha_var=0, alpha_skew=0, alpha_outliers=0, alpha_cat=0, max_new_cat=1, new_cat=False):

  """
  Induces drift synthetically (statistically) to the dataset given.

  df: DataFrame
  features: List of columns in df to which drift needs to be induced, 
  alpha_mean: Alpha value to shift mean
  alpha_var: Alpha value to shift variance
  alpha_skew: Alpha value to shift skewness
  alpha_outliers: Alpha value for inducing outliers
  alpha_cat: Proportion od Categorical that gets perturbed
  max_new_cat: Number of new categories to be added to each column
  new_cat: Boolean, tells if new categories should be added or not

  If a function is called without setting any parameters, dataframe is returned directly as default values for all params is 0 (or None).

  """

  print(f"Parameters were set to: \nMean shift:{alpha_mean} Variance shift:{alpha_var} Skewness shift:{alpha_skew} Outliers shift:{alpha_outliers} \nCategorical shift: {alpha_cat} Max New category: {max_new_cat} Add New Categories?: {new_cat}")

  df = data.copy()

  if features is None or not any([alpha_mean, alpha_outliers, alpha_skew, alpha_var, alpha_cat, max_new_cat, 0]):
    print("No drift induced")
  else:

    cat_ft, num_ft = determine_dtype_ft(df)
    
    for i in features:
      if i in num_ft:
        df[i] = shift_mean(df[i], alpha=alpha_mean)
        df[i] = shift_var(df[i], alpha=alpha_var)
        df[i] = shift_skewness(df[i], alpha=alpha_skew)
        df[i] = induce_outliers(df[i], alpha=alpha_outliers)
      else:
        df[i] = categorical_shift(df[i], alpha=alpha_cat, new_cat=new_cat, max_new_cat=max_new_cat)

  return df


def get_prompt(prompt, production_date, prod_summary, performance_drift, data_drift, data_quality, prediction_drift, analysis_type):
    
    if analysis_type == "Output Features/Performance Related":
    
        message = f"""Information about current production run on {production_date}:
    
    An overall summary of the data we got on this production run is:
    {prod_summary}
    
    On performing analyses, we get the following results:
    
    For Performance Drift Analysis results are: 
    {performance_drift}
    
    For Prediction Drift Analysis results are: 
    {prediction_drift}
    
    Prompt: {prompt}
    
    Be very brief and informative. Also, make sure to follow the instructions provided by the system, DO NOT make your own answers, go through the results and answer ONLY from it.
    The prompts are asked based on the current analysis selection, if you think the current prompt can be answered by other analysis selection, please suggest.
    
    Example: "The current analysis selected is Data Drift Analysis, but the prompt can be answered by Performance Drift Analysis such as model performance. Please suggest."
   
    """
    
    if analysis_type == "Input Features Related":
        message = f"""Information about current production run on {production_date}:
    
    An overall summary of the data we got on this production run is:
    {prod_summary}
    
    On performing analyses, we get the following results:
    
    For Data Drift Analysis results are: 
    {data_drift}
    
    For Data Quality Analysis results are: 
    {data_quality}
    
    Prompt: {prompt}
    
    Be very brief and informative. Also, make sure to follow the instructions provided by the system, DO NOT make your own answers, go through the results and answer ONLY from it.
    The prompts are asked based on the current analysis selection, if you think the current prompt can be answered by other analysis selection, please suggest.
    
    Example: "The current analysis selected is Data Drift Analysis, but the prompt can be answered by Performance Drift Analysis such as model performance. Please suggest."
    """
    
    return message

def get_domain_knowledge(model_name):
    
    with open("./pages/domain_knowledge.json", "r") as m:
        domain_dict = json.load(m)
    
    return domain_dict[model_name]

def create_ollama_model(model_name, model_type, domain_knowledge, baseline_stats, output_type=None):
    
    llm_name = "_".join(model_name.lower().split())
    
    template = """{{- if .Messages }}
{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}
{{- if and (eq (len (slice $.Messages $index)) 1) $.Tools }}[AVAILABLE_TOOLS] {{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST] {{ if and $.System (eq (len (slice $.Messages $index)) 1) }}{{ $.System }}

{{ end }}{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }} {{ .Content }}
{{- else if .ToolCalls }}[TOOL_CALLS] [
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}]
{{- end }}</s>
{{- else if eq .Role "tool" }}[TOOL_RESULTS] {"content": {{ .Content }}} [/TOOL_RESULTS]
{{- end }}
{{- end }}
{{- else }}[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}[/INST]
{{- end }} {{ .Response }}
{{- if .Response }}</s>
{{- end }}"""
    
    
    if model_type in ["Classification", "Regression"]:
        
        system = f"""You are a very smart, knowledgeable, and helpful assistant that answers questions related to model degradation issues, metrics, and data of a {model_name} application, which is a {model_type} task."

    You are part of a root cause analysis application where machine learning models are deployed and assessed against 5 analyses types:
    1) Performance Drift Analysis: Compare model performance metrics (e.g., accuracy, precision, recall) on current production run with ground truths
    2) Prediction Drift Analysis: Perform tests like KS-test, Chi-square test, PSI and JS on output label to analyze drift
    3) Data Drift Analysis: Perform tests like KS-test, Chi-square test, PSI and JS on input data to analyze drift
    4) Data Quality Analysis: Analyze data quality metrics such as missing values, duplicates, business rules, datatype mismatches and outliers
    5) Model Explanations/Interpretations: Use SHAP to provide explanations for each instance in the data using the model

    You'll be provided domain knowledge about the model, its baseline data, and production data (based on the production date).

    Domain Knowledge:
    {domain_knowledge}

    Baseline Information:
    {baseline_stats}
        
    Instructions:
    1) DO NOT provide false information. Answer only based on available data.
    2) Provide ONLY accurate information about metrics or clinical jargon.
    3) Keep answers concise (≤30 words).
    4) Always be polite, ethical, and safe. NO harmful, illegal, or offensive responses.
    5) If you don’t know the answer, admit it instead of guessing.
    6) Suggest potential root causes for model degradation while following all other guidelines.

    Your name is {llm_name}_mistralLLM.
    You are a large language model that is very helpful and knowledgeable about the {model_name} application.
    You'll be given information about the analysis results for the production run in the prompt."""
    
    if model_type == "Natural Language Processing (NLP)":
        
        system = f"""You are a very smart, knowledgeable, and helpful assistant that answers questions related to model degradation issues, metrics, and data of a {model_name} application, which is a {model_type} task, giving outputs as {output_type}"

    You are a Computer Vision model, the input for the model is an image and features are extracted from the image.
    
    You are part of a root cause analysis application where machine learning models are deployed and assessed against 5 analyses types:
    1) Performance Drift Analysis: Compare model performance metrics (e.g., accuracy, precision, recall) on current production run with ground truths
    2) Prediction Drift Analysis: Perform tests like Chi-square test, PSI and JS on output label to analyze drift
    3) Data Drift Analysis: Perform tests like KS-test, Chi-square test, PSI and JS on features extracted from input images to analyze drift
    4) Data Quality Analysis: Analyze data quality metrics such as sharpness, brightness, noise, resolution, size, and anomalies
    5) Model Explanations/Interpretations: Use LIME to provide explanations for each instance in the data using the model

    You'll be provided domain knowledge about the model, its baseline data, and production data (based on the production date).

    Domain Knowledge:
    {domain_knowledge}

    Baseline Information:
    {baseline_stats}
        
    Instructions:
    1) DO NOT provide false information. Answer only based on available data.
    2) Provide ONLY accurate information about metrics or clinical jargon.
    3) Keep answers concise (≤30 words).
    4) Always be polite, ethical, and safe. NO harmful, illegal, or offensive responses.
    5) If you don’t know the answer, admit it instead of guessing.
    6) Suggest potential root causes for model degradation while following all other guidelines.

    Your name is {llm_name}_mistralLLM.
    You are a large language model that is very helpful and knowledgeable about the {model_name} application.
    You'll be given information about the analysis results for the production run in the prompt."""
    
    
    
    if model_type == "Computer Vision (CV)":
        
        system = f"""You are a very smart, knowledgeable, and helpful assistant that answers questions related to model degradation issues, metrics, and data of a {model_name} application, which is a {model_type}, Image Classification task."

    You are a Computer Vision model, the input for the model is an image and features are extracted from the image.
    
    You are part of a root cause analysis application where machine learning models are deployed and assessed against 5 analyses types:
    1) Performance Drift Analysis: Compare model performance metrics (e.g., accuracy, precision, recall) on current production run with ground truths
    2) Prediction Drift Analysis: Perform tests like Chi-square test, PSI and JS on output label to analyze drift
    3) Data Drift Analysis: Perform tests like KS-test, Chi-square test, PSI and JS on features extracted from input images to analyze drift
    4) Data Quality Analysis: Analyze data quality metrics such as sharpness, brightness, noise, resolution, size, and anomalies
    5) Model Explanations/Interpretations: Use LIME to provide explanations for each instance in the data using the model

    You'll be provided domain knowledge about the model, its baseline data, and production data (based on the production date).

    Domain Knowledge:
    {domain_knowledge}

    Baseline Information:
    {baseline_stats}
        
    Instructions:
    1) DO NOT provide false information. Answer only based on available data.
    2) Provide ONLY accurate information about metrics or clinical jargon.
    3) Keep answers concise (≤30 words).
    4) Always be polite, ethical, and safe. NO harmful, illegal, or offensive responses.
    5) If you don’t know the answer, admit it instead of guessing.
    6) Suggest potential root causes for model degradation while following all other guidelines.

    Your name is {llm_name}_mistralLLM.
    You are a large language model that is very helpful and knowledgeable about the {model_name} application.
    You'll be given information about the analysis results for the production run in the prompt."""
    
    
    license = """                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License."""


    
    
    modelfile = f'''
    # Modelfile generated by "ollama show"
# To build a new Modelfile based on this, replace FROM with:
# FROM mistral:latest

FROM C:\\Users\\Akshat Mittu\\.ollama\\models\\blobs\\sha256-ff82381e2bea77d91c1b824c7afb83f6fb73e9f7de9dda631bcdbca564aa5435
TEMPLATE """{template}"""

PARAMETER stop [INST]
PARAMETER stop [/INST]

SYSTEM """{system}"""
LICENSE """{license}"""
'''
    with open(f"./pages/models/{model_name}/{llm_name}.modelfile", "w", encoding="utf-8") as m:
        m.write(modelfile)
    
    os.system(f"ollama create {llm_name}_mistralLLM --file {llm_name}.modelfile")
    
    
    return "Success"


def get_statistics(column, column_type, drift_results=None, baseline=False, domain_knowledge=None):
    
    stat = dict()
    
    # print(drift_results)
    
    if column_type == "categorical":
        
        stat['mode'] = str(column.mode()[0])
        stat['value_counts'] = column.value_counts().to_dict()
        
        if baseline == False:
            stat['type'] = str(column.dtype)
            stat['Chi Square statistic'] = drift_results[column.name]['chi']
            stat['Chi Square p-value'] = drift_results[column.name]['chi_p-value']
            stat['Drift status given by Chi Square'] = True if drift_results[column.name]['chi_status'] == True else False
            stat['PSI statistic'] = drift_results[column.name]['psi_value']
            stat['Threshold used in PSI'] = drift_results[column.name]['psi_threshold']
            stat['Drift status given by PSI'] = True if drift_results[column.name]['psi_status'] == True else False
            stat['JS Distance statistic'] = drift_results[column.name]['jsd_value']
            stat['Threshold used in JS Distance'] = drift_results[column.name]['js_threshold']
            stat['Drift status given by JS Distance'] = True if drift_results[column.name]['js_status'] == True else False
            status_list = [stat['Drift status given by Chi Square'], stat['Drift status given by PSI'], stat['Drift status given by JS Distance']]
            stat['Final drift status'] = True if status_list.count(True) >= 2 else False
        
    if column_type == "numerical":
        
        q1 = np.nanpercentile(column,25)
        q3 = np.nanpercentile(column,75)
        iqr_values = q3 - q1
        stat['mean'] = float(round(np.nanmean(column),2))
        stat['median'] = float(round(np.nanmedian(column),2))
        stat['std'] = float(round(np.nanstd(column),2))
        stat['min'] = float(round(np.nanmin(column),2))
        stat['max'] = float(round(np.nanmax(column),2))
        stat['iqr'] = float(iqr_values)
        
        if baseline == False:
            stat['type'] = str(column.dtype)
            stat['KS-Test statistic'] = drift_results[column.name]['ks-stat']
            stat['KS-Test p-value'] = drift_results[column.name]['ks_p-value']
            stat['Drift status given by KS-Test'] = True if drift_results[column.name]['ks_status'] == True else False
            stat['PSI statistic'] = drift_results[column.name]['psi_value']
            stat['Threshold used in PSI'] = drift_results[column.name]['psi_threshold']
            stat['Drift status given by PSI'] = True if drift_results[column.name]['psi_status'] == True else False
            stat['JS Distance statistic'] = drift_results[column.name]['jsd_value']
            stat['Threshold used in JS Distance'] = drift_results[column.name]['js_threshold']
            stat['Drift status given by JS Distance'] = True if drift_results[column.name]['js_status'] == True else False
            status_list = [stat['Drift status given by KS-Test'], stat['Drift status given by PSI'], stat['Drift status given by JS Distance']]
            stat['Final drift status'] = True if status_list.count(True) >= 2 else False
        
    return stat


def psi(baseline, production, categorical_ft=None, bins=None,threshold=0.2):
    """
    Calculate the Population Stability Index (PSI) between a baseline dataset and a production dataset.
    
    Parameters:
    baseline (numpy.array): The production dataset, representing the baseline distribution.
    production (numpy.array): The baseline dataset, representing the distribution to compare against the production.
    bins (int, optional): The number of bins to use for the histograms. If set to None, Doane's formula will be used to calculate the number of bins. Default is None.
    threshold: The value which decides the presence of drift.
    Returns:
    float: The calculated PSI value. A higher value indicates greater divergence between the two distributions.
    """
    if categorical_ft is None:
        # Get the full dataset
        
        full_dataset = np.concatenate((baseline, production))
        encoder = LabelEncoder()
        encoder.fit(full_dataset)
        baseline_enc = encoder.transform(baseline)
        production_enc = encoder.transform(production)
        full_dataset_enc = encoder.transform(full_dataset)
    
        # If bins is not parametrized, use Doane's formula for calculating number of bins
        if bins is None:
            _, bin_edges = np.histogram(full_dataset_enc, bins="doane")
        else:  # If number of bins is specified
            bin_edges = np.linspace(min(min(baseline_enc), min(production_enc)), max(max(baseline_enc), max(production_enc)), bins + 1)
    
        # Calculate the histogram for each dataset
        baseline_hist, _ = np.histogram(baseline_enc, bins=bin_edges)
        production_hist, _ = np.histogram(production_enc, bins=bin_edges)
    
        # Convert histograms to proportions
        baseline_proportions = baseline_hist / np.sum(baseline_hist)
        production_proportions = production_hist / np.sum(production_hist)
    
        # Replace zeroes to avoid division by zero or log of zero errors
        baseline_proportions = np.where(baseline_proportions == 0, 1e-6, baseline_proportions)
        production_proportions = np.where(production_proportions == 0, 1e-6, production_proportions)
    
        # Calculate PSI
        psi_values = (baseline_proportions - production_proportions) * np.log(baseline_proportions / production_proportions)
        psi = np.sum(psi_values)

        result_dict = {
            "psi_value" : round(psi,2),
            "psi_threshold" : threshold,
            "psi_status": bool(psi>threshold)
        }

        baseline_dict = dict(baseline.value_counts())
        prod_dict = dict(production.value_counts())


        baseline_unique, prod_unique = list(baseline_dict.keys()), list(prod_dict.keys())
        features = list(set(baseline_unique + prod_unique))
        
        baseline_props = {j:baseline_dict[j] if j in baseline_unique else 0 for j in features} 
        prod_props = {j:prod_dict[j] if j in prod_unique else 0 for j in features}

        # print(baseline_props, prod_props)

        figure = go.Figure()
        figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
        figure.add_traces(go.Bar(x=list(prod_props.keys()), y=list(prod_props.values()), text=list(prod_props.values()), name="Production"))

        figure.update_layout(
                        xaxis_title=f'Categories in Target',
                        yaxis_title='Category Value Counts ',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=500)
        figure.update_xaxes(tickfont={"size":14, "color":"black"})
        figure.update_yaxes(tickfont={"size":14, "color":"black"})
        figure.update_layout(barmode='group')
        
        return result_dict,figure
        
    else:
        result_dict = dict()
        figures = dict()
        for i in categorical_ft:
            full_dataset = np.concatenate((baseline[i], production[i]))
            encoder = LabelEncoder()
            encoder.fit(full_dataset)
            baseline_enc = encoder.transform(baseline[i])
            production_enc = encoder.transform(production[i])
            full_dataset_enc = encoder.transform(full_dataset)
            # print(encoder.classes_)
            if bins is None:
                _, bin_edges = np.histogram(full_dataset_enc, bins="doane")
            else:  # If number of bins is specified
                bin_edges = np.linspace(min(min(baseline_enc), min(production_enc)), max(max(baseline_enc), max(production_enc)), bins + 1)
        
            # Calculate the histogram for each dataset
            production_hist, _ = np.histogram(baseline_enc, bins=bin_edges)
            baseline_hist, _ = np.histogram(production_enc, bins=bin_edges)
        
            # Convert histograms to proportions
            production_proportions = production_hist / np.sum(production_hist)
            baseline_proportions = baseline_hist / np.sum(baseline_hist)
        
            # Replace zeroes to avoid division by zero or log of zero errors
            baseline_proportions = np.where(baseline_proportions == 0, 1e-6, baseline_proportions)
            production_proportions = np.where(production_proportions == 0, 1e-6, production_proportions)
        
            # Calculate PSI
            psi_values = (baseline_proportions - production_proportions) * np.log(baseline_proportions / production_proportions)
            psi = np.sum(psi_values)
            result_dict[i] = {"psi_value":round(psi,2), "psi_threshold": threshold, "psi_status": bool(psi>threshold)}


            baseline_dict = dict(baseline[i].value_counts())
            prod_dict = dict(production[i].value_counts())


            baseline_unique, prod_unique = list(baseline_dict.keys()), list(prod_dict.keys())
            features = list(set(baseline_unique + prod_unique))

            baseline_props = {j:baseline_dict[j] if j in baseline_unique else 0 for j in features} 
            prod_props = {j:prod_dict[j] if j in prod_unique else 0 for j in features}

            # print(baseline_props, prod_props)

            figure = go.Figure()
            figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
            figure.add_traces(go.Bar(x=list(prod_props.keys()), y=list(prod_props.values()), text=list(prod_props.values()), name="Production"))

            figure.update_layout(
                            xaxis_title=f'Categories in {i}',
                            yaxis_title='Category Value Counts ',
                            # title_font={"size": 20},
                            xaxis_title_font={"size":16, "color":"black"},
                            yaxis_title_font={"size":16, "color":"black"},
                            width=1080,
                            height=500)
            figure.update_xaxes(tickfont={"size":14, "color":"black"})
            figure.update_yaxes(tickfont={"size":14, "color":"black"})
            figure.update_layout(barmode='group')

            figures[i] = figure
            
        return result_dict,figures

def get_baseline_statistics(model_name, baseline_data, model_type, output_type=None, target_file=None):
    
    completeness, _, WO_nan, w_nan, missing_data = data_completeness(baseline_data.drop(["target"],axis=1))
    uniqueness_score,u_s, _, _, no_of_rows_wo_dup = data_uniqueness(baseline_data.drop(["target"],axis=1))
    outliers_index = outliers_for_num_data(baseline_data.drop('target', axis=1))
    
    if model_type == "Classification" :
        output_features = get_statistics(baseline_data["target"],"categorical",baseline=True)
    
    if model_type == "Regression":
        output_features = get_statistics(baseline_data["target"],"numerical",baseline=True)
        
    if model_type == "Natural Language Processing (NLP)":
        words = get_words(baseline_data)
        misspelled_words, percentage, _ = check_spelling(words)
        
        if output_type == "Text":

            words_target = get_words(baseline_data, 'target')
            misspell, per_tar, _ = check_spelling(words_target)


            inter_target_dict = {'length': get_statistics(target_file['length'], "numerical", baseline=True),
                                 'readability': get_statistics(target_file['readability'], "numerical", baseline=True)}
            output_features = inter_target_dict

            output_features.update({'Number of words in target column': len(words_target), 
                                   "Number of misspelled words in target column":misspell,
                   "Percentage words misspelled": per_tar})

        if output_type == "Classification":
            output_features = get_statistics(baseline_data["target"],"categorical",baseline=True)
        
        pass
        
    if model_type == "Computer Vision (CV)":
        output_features = get_statistics(baseline_data["target"],"categorical",baseline=True)


    summary = {
        "Number of rows" : float(len(baseline_data)),
        "Number of rows without NaN" : float(WO_nan),
        "Number of rows with NaN" : float(w_nan),
        "Completeness score" : float(completeness),
        "Uniqueness score" : float(uniqueness_score),
        "Number of rows without duplicates" : float(no_of_rows_wo_dup),
        "Number of rows with duplicates" : float(len(baseline_data) - no_of_rows_wo_dup)
    }
    
    summary.update({'Number of words in text column': len(words), "Number of misspelled words in text column":misspelled_words, 
                   "Percentage words misspelled": percentage}) if model_type == 'Natural Language Processing (NLP)' else summary
   
    cols = baseline_data.drop(["target"],axis=1).columns if 'text' not in baseline_data.columns else baseline_data.drop(["target", "text"],axis=1).columns
    input_features = dict()
    input_u_s = {u_s['Feature'][i] : round(u_s['Value'][i],2) for i in range(len(u_s['Feature']))}
    outliers = {i : float(len(outliers_index[i])) for i in outliers_index}
    categorical_ft, numerical_ft = determine_dtype_ft(baseline_data.drop(["target"],axis = 1))


    for i in cols:
        input_features[i] = {
                "Number of missing values" : float(dict(missing_data)[i]),
                "Uniqueness score" : float(input_u_s[i]),
                "Number of outliers" : float(outliers[i]),
                }

        if i in categorical_ft:
            input_features[i].update(get_statistics(baseline_data[i], "categorical",baseline=True))
        if i in numerical_ft:
            input_features[i].update(get_statistics(baseline_data[i], "numerical",baseline=True))
    
    results_dict = {model_name : {"Baseline Data Summary": summary, "Input Feature Details": input_features, "Target": output_features}}
    
    return results_dict

def js_test(baseline, production, features=None,threshold=0.2):
    """
    Calculate the Jensen-Shannon divergence (JS) metric between a baseline dataset and a production dataset.
    
    Parameters:
    baseline (numpy.array): The production dataset, representing the baseline distribution.
    production (numpy.array): The baseline dataset, representing the distribution to compare against the production.
    Returns:
    float: The calculated JSD value. A higher value indicates greater divergence between the two distributions.
    """
    if features is None:

        baseline_dict = dict(baseline.value_counts())
        prod_dict = dict(production.value_counts())


        baseline_unique, prod_unique = list(baseline_dict.keys()), list(prod_dict.keys())
        features = list(set(baseline_unique + prod_unique))
        
        baseline_props = {j:baseline_dict[j] if j in baseline_unique else 0 for j in features} 
        prod_props = {j:prod_dict[j] if j in prod_unique else 0 for j in features}

        baseline_probs = [i/len(baseline) for i in baseline_props.values()]
        prod_probs = [i/len(production) for i in prod_props.values()]

        jsd = jensenshannon(baseline_probs,prod_probs)
        
        result_dict = {
            "jsd_value" : round(jsd,2),
            "js_threshold" : threshold,
            "js_status": bool(jsd>threshold)
        }

        figure = go.Figure()
        figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
        figure.add_traces(go.Bar(x=list(prod_props.keys()), y=list(prod_props.values()), text=list(prod_props.values()), name="Production"))

        figure.update_layout(
                        xaxis_title=f'Categories in Target',
                        yaxis_title='Category Value Counts ',
                        # title_font={"size": 20},
                        xaxis_title_font={"size":16, "color":"black"},
                        yaxis_title_font={"size":16, "color":"black"},
                        width=1080,
                        height=500)
        figure.update_xaxes(tickfont={"size":14, "color":"black"})
        figure.update_yaxes(tickfont={"size":14, "color":"black"})
        figure.update_layout(barmode='group')
        
        return result_dict,figure
        
    else:
        result_dict = dict()
        figures = dict()
        for i in features:
            
            baseline_dict = dict(baseline[i].value_counts())
            prod_dict = dict(production[i].value_counts())


            baseline_unique, prod_unique = list(baseline_dict.keys()), list(prod_dict.keys())
            features = list(set(baseline_unique + prod_unique))

            baseline_props = {j:baseline_dict[j] if j in baseline_unique else 0 for j in features} 
            prod_props = {j:prod_dict[j] if j in prod_unique else 0 for j in features}

            baseline_probs = [j/len(baseline) for j in baseline_props.values()]
            prod_probs = [j/len(production) for j in prod_props.values()]
    
            jsd = jensenshannon(baseline_probs,prod_probs)
            
            result_dict[i] = {
                "jsd_value" : round(jsd,2),
                "js_threshold" : threshold,
                "js_status": bool(jsd>threshold)
            }

            # print(baseline_props, prod_props)

            figure = go.Figure()
            figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
            figure.add_traces(go.Bar(x=list(prod_props.keys()), y=list(prod_props.values()), text=list(prod_props.values()), name="Production"))

            figure.update_layout(
                            xaxis_title=f'Categories in {i}',
                            yaxis_title='Category Value Counts ',
                            # title_font={"size": 20},
                            xaxis_title_font={"size":16, "color":"black"},
                            yaxis_title_font={"size":16, "color":"black"},
                            width=1080,
                            height=500)
            figure.update_xaxes(tickfont={"size":14, "color":"black"})
            figure.update_yaxes(tickfont={"size":14, "color":"black"})
            figure.update_layout(barmode='group')

            figures[i] = figure
        return result_dict,figures
 
def psi_num(baseline, production, num_ft=None, nlp=False, threshold=0.05, num_bins=10, b_name="Baseline Data"):
    if nlp == True:
        x = "Frequency"
    else:
        x = "Target Values"

    result_dict = dict()
    figures = dict()

    if num_ft is None or nlp:
        bins = np.linspace(min(min(baseline), min(production)), max(max(baseline), max(production)), num_bins + 1)

        # Calculate the bin percentage distribution for both datasets
        baseline_binned = pd.cut(baseline, bins, include_lowest=True)
        production_binned = pd.cut(production, bins, include_lowest=True)

        # Calculate the percentage of data points in each bin
        baseline_pct = baseline_binned.value_counts(normalize=True).round(2)
        production_pct = production_binned.value_counts(normalize=True).round(2)

        # Ensure all bins are represented in both distributions
        all_bins = baseline_binned.cat.categories
        baseline_pct = baseline_pct.reindex(all_bins, fill_value=0)
        production_pct = production_pct.reindex(all_bins, fill_value=0)

        # Calculate PSI for each bin, avoiding log(0) by adding epsilon
        epsilon = 1e-10
        psi_values = (baseline_pct - production_pct) * np.log((baseline_pct + epsilon) / (production_pct + epsilon))

        # Sum up PSI values to get the final PSI
        psi_value = psi_values.sum()

        # Store the results
        result_dict["psi_value"] = round(psi_value, 2)
        result_dict["psi_threshold"] = threshold
        result_dict["psi_status"] = bool(psi_value > threshold)

        # Prepare the bar plot data
        baseline_dict = dict(baseline_pct)
        prod_dict = dict(production_pct)
        all_categories = list(set(baseline_dict.keys()).union(set(prod_dict.keys())))

        baseline_props = {cat: baseline_dict.get(cat, 0) for cat in all_categories}
        prod_props = {cat: prod_dict.get(cat, 0) for cat in all_categories}

        # Convert the intervals to strings and sort by their lower bounds
        baseline_props = {str(k): v for k, v in baseline_props.items()}
        prod_props = {str(k): v for k, v in prod_props.items()}

        # Sort by the lower bound of the intervals
        baseline_props = dict(sorted(baseline_props.items(), key=lambda x: float(x[0].split(',')[0].replace('(', ''))))
        prod_props = dict(sorted(prod_props.items(), key=lambda x: float(x[0].split(',')[0].replace('(', ''))))

        # Create the plot
        figure = go.Figure()
        figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
        figure.add_traces(go.Bar(x=list(prod_props.keys()), y=list(prod_props.values()), text=list(prod_props.values()), name="Production"))

        figure.update_layout(
            xaxis_title=f'Bins in {x}',
            yaxis_title='Bin Value Counts',
            width=1080,
            height=500
        )
        figure.update_xaxes(tickfont={"size": 14, "color": "black"})
        figure.update_yaxes(tickfont={"size": 14, "color": "black"})
        figure.update_layout(barmode='group')

        return result_dict, figure

    else:
        for feature in num_ft:
            bins = np.linspace(min(baseline[feature].min(), production[feature].min()), 
                               max(baseline[feature].max(), production[feature].max()), 
                               num_bins + 1)

            # Calculate the bin percentage distribution for both datasets
            baseline_binned = pd.cut(baseline[feature], bins, include_lowest=True)
            production_binned = pd.cut(production[feature], bins, include_lowest=True)

            # Calculate the percentage of data points in each bin
            baseline_pct = baseline_binned.value_counts(normalize=True).round(2)
            production_pct = production_binned.value_counts(normalize=True).round(2)

            # Ensure all bins are represented in both distributions
            all_bins = baseline_binned.cat.categories
            baseline_pct = baseline_pct.reindex(all_bins, fill_value=0)
            production_pct = production_pct.reindex(all_bins, fill_value=0)

            # print(baseline_pct,production_pct)

            # Calculate PSI for each bin, avoiding log(0) by adding epsilon
            epsilon = 1e-10
            psi_values = (baseline_pct - production_pct) * np.log((baseline_pct) / (production_pct))

            # Sum up PSI values to get the final PSI
            psi_value = psi_values.sum()

            result_dict[feature] = {
                "psi_value": round(psi_value, 2),
                "psi_threshold": threshold,
                "psi_status": bool(psi_value > threshold)
            }

            # Prepare the bar plot data
            baseline_dict = dict(baseline_pct)
            prod_dict = dict(production_pct)
            all_categories = list(set(baseline_dict.keys()).union(set(prod_dict.keys())))

            baseline_props = {cat: baseline_dict.get(cat, 0) for cat in all_categories}
            prod_props = {cat: prod_dict.get(cat, 0) for cat in all_categories}

            # Convert the intervals to strings and sort by their lower bounds
            baseline_props = {str(k): v for k, v in baseline_props.items()}
            prod_props = {str(k): v for k, v in prod_props.items()}

            # Sort by the lower bound of the intervals
            baseline_props = dict(sorted(baseline_props.items(), key=lambda x: float(x[0].split(',')[0].replace('(', ''))))
            prod_props = dict(sorted(prod_props.items(), key=lambda x: float(x[0].split(',')[0].replace('(', ''))))

            # Create the plot
            figure = go.Figure()
            figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
            figure.add_traces(go.Bar(x=list(prod_props.keys()), y=list(prod_props.values()), text=list(prod_props.values()), name="Production"))

            figure.update_layout(
                xaxis_title=f'Bins in {feature}',
                yaxis_title='Bin Value Counts',
                width=1080,
                height=500
            )
            figure.update_xaxes(tickfont={"size": 14, "color": "black"})
            figure.update_yaxes(tickfont={"size": 14, "color": "black"})
            figure.update_layout(barmode='group')

            figures[feature] = figure

        return result_dict, figures
               
def js_num(baseline, production, num_ft=None, nlp=False, threshold=0.1, num_bins=10):
    if nlp:
        x = "Frequency"
    else:
        x = "Target Values"

    result_dict = dict()
    figures = dict()

    if num_ft is None or nlp:
        bins = np.linspace(min(min(baseline), min(production)), max(max(baseline), max(production)), num_bins + 1)
        baseline_binned = pd.cut(baseline, bins, include_lowest=True)
        production_binned = pd.cut(production, bins, include_lowest=True)

        baseline_pct = baseline_binned.value_counts(normalize=True).reindex(baseline_binned.cat.categories, fill_value=0).round(2)
        production_pct = production_binned.value_counts(normalize=True).reindex(production_binned.cat.categories, fill_value=0).round(2)

        baseline_unique, prod_unique = list(baseline_pct.keys()), list(production_pct.keys())
        features = list(set(baseline_unique + prod_unique))

        baseline_props = {j:baseline_pct[j] if j in baseline_unique else 0 for j in features} 
        prod_props = {j:production_pct[j] if j in prod_unique else 0 for j in features}

        baseline_probs = [j/len(baseline) for j in baseline_props.values()]
        prod_probs = [j/len(production) for j in prod_props.values()]
    
        jsd = jensenshannon(baseline_probs,prod_probs)**2

        result_dict["jsd_value"] = round(jsd, 2)
        result_dict["js_threshold"] = threshold
        result_dict["js_status"] = bool(jsd > threshold)

        baseline_props = {str(k): v for k, v in baseline_pct.items()}
        production_props = {str(k): v for k, v in production_pct.items()}

        baseline_props = dict(sorted(baseline_props.items(), key=lambda x: float(x[0].split(',')[0].replace('(', ''))))
        production_props = dict(sorted(production_props.items(), key=lambda x: float(x[0].split(',')[0].replace('(', ''))))

        figure = go.Figure()
        figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
        figure.add_traces(go.Bar(x=list(production_props.keys()), y=list(production_props.values()), text=list(production_props.values()), name="Production"))

        figure.update_layout(
            xaxis_title=f'Bins in {x}',
            yaxis_title=f'{x} Proportions',
            width=1080,
            height=500,
            barmode='group'
        )

        return result_dict, figure

    else:
        for feature in num_ft:
            bins = np.linspace(min(baseline[feature].min(), production[feature].min()), 
                               max(baseline[feature].max(), production[feature].max()), 
                               num_bins + 1)

            baseline_binned = pd.cut(baseline[feature], bins, include_lowest=True)
            production_binned = pd.cut(production[feature], bins, include_lowest=True)

            baseline_pct = baseline_binned.value_counts(normalize=True).reindex(baseline_binned.cat.categories, fill_value=0).round(2)
            production_pct = production_binned.value_counts(normalize=True).reindex(production_binned.cat.categories, fill_value=0).round(2)

            baseline_unique, prod_unique = list(baseline_pct.keys()), list(production_pct.keys())
            features = list(set(baseline_unique + prod_unique))

            baseline_props = {j:baseline_pct[j] if j in baseline_unique else 0 for j in features} 
            prod_props = {j:production_pct[j] if j in prod_unique else 0 for j in features}

            baseline_probs = [j/len(baseline) for j in baseline_props.values()]
            prod_probs = [j/len(production) for j in prod_props.values()]
    
            jsd = jensenshannon(baseline_probs,prod_probs)**2

            result_dict[feature] = {
                "jsd_value": round(jsd, 2),
                "js_threshold": threshold,
                "js_status": bool(jsd > threshold)
            }

            baseline_props = {str(k): v for k, v in baseline_pct.items()}
            production_props = {str(k): v for k, v in production_pct.items()}

            baseline_props = dict(sorted(baseline_props.items(), key=lambda x: float(x[0].split(',')[0].replace('(', ''))))
            production_props = dict(sorted(production_props.items(), key=lambda x: float(x[0].split(',')[0].replace('(', ''))))

            figure = go.Figure()
            figure.add_traces(go.Bar(x=list(baseline_props.keys()), y=list(baseline_props.values()), text=list(baseline_props.values()), name="Baseline"))
            figure.add_traces(go.Bar(x=list(production_props.keys()), y=list(production_props.values()), text=list(production_props.values()), name="Production"))

            figure.update_layout(
                xaxis_title=f'Categories in {feature}',
                yaxis_title=f'{feature} Value Proportions',
                width=1080,
                height=500,
                barmode='group'
            )

            figures[feature] = figure

        return result_dict, figures
    

def separated_results(result_dict, model_name, prod_date, num_cols, cat_cols, model_type):
    
    results = result_dict[model_name][prod_date]
    data_drift, data_quality = dict(), dict()
    input_details = results['Input Feature Details']
    
    # if model_type == "Natural Language Processing (NLP)":
    #     new_cat_cols = [i for i in cat_cols if i != 'text']
    #     print(f"Here:{new_cat_cols}")
    # else:
    #     new_cat_cols = cat_cols
    #     print()
    
    # if model_type == "Computer Vision (CV)":
    #     cat_cols.append('Anomaly')
    #     new_cat_cols = [i for i in cat_cols if i != 'paths']
    #     num_cols.remove("Anomaly")
        
    # else:
    #     new_cat_cols = cat_cols
    
    # print(f"Num cols: {num_cols}")
    # print(f"Cat cols: {new_cat_cols}")
    
    cat_cols.remove("text") if "text" in cat_cols else None
    cat_cols.remove("paths") if "paths" in cat_cols else None
    if "Anomaly" in num_cols:
        num_cols.remove("Anomaly")
        cat_cols.append("Anomaly")
    
    
    cols = num_cols + cat_cols
    print(cols)
    # Performance Dict
    performance_dict = results['Production Data Summary']['Metrics']
    performance_dict['Drifting Performance Metrics'] = results['Production Data Summary']['Drifting Metrics']
    
    # Data Drift and Data Quality Dicts
    for i in cols:
        
        print(i)
        
        if i in num_cols:
            
            data_drift[i] = {'KS-Test statistic': input_details[i]['KS-Test statistic'],
                            'KS-Test p-value': input_details[i]['KS-Test p-value'],
                            'Drift status given by KS-Test': input_details[i]['Drift status given by KS-Test'],
                            'PSI statistic': input_details[i]['PSI statistic'],
                            'Threshold used in PSI': input_details[i]['Threshold used in PSI'],
                            'Drift status given by PSI': input_details[i]['Drift status given by PSI'],
                            'JS Distance statistic': input_details[i]['JS Distance statistic'],
                            'Threshold used in JS Distance': input_details[i]['Threshold used in JS Distance'],
                            'Drift status given by JS Distance': input_details[i]['Drift status given by JS Distance'],
                            'Final drift status by voting': input_details[i]['Final drift status']}
            
           # Number of duplicates, completeness score, number of invalid points, validity score
        
            data_quality[i] = {'Number of missing values': input_details[i]['Number of missing values'] if model_type != "Computer Vision (CV)" else 0,
                          'Uniqueness Score': input_details[i]['Uniqueness Score'] if model_type != "Computer Vision (CV)" else 100,
                          'Number of datatype mismatches': input_details[i]['Number of datatype mismatches'] if model_type != "Computer Vision (CV)" else 0,
                          'Number of outliers': input_details[i]['Number of outliers'] if model_type != "Computer Vision (CV)" else 0,
                          'Mean': input_details[i]['mean'],
                          'Datatype': input_details[i]['type'],
                          'Median': input_details[i]['median'],
                          'Standard Deviation': input_details[i]['std'],
                          'Min value': input_details[i]['min'],
                          'Max value': input_details[i]['max'],
                          'Inter quartile Range': input_details[i]['iqr']
                          }
            
        if i in cat_cols:
            
            data_drift[i] = {'Chi Square statistic': input_details[i]['Chi Square statistic'],
                            'Chi Square p-value': input_details[i]['Chi Square p-value'],
                            'Drift status given by Chi Square': input_details[i]['Drift status given by Chi Square'],
                            'PSI statistic': input_details[i]['PSI statistic'],
                            'Threshold used in PSI': input_details[i]['Threshold used in PSI'],
                            'Drift status given by PSI': input_details[i]['Drift status given by PSI'],
                            'JS Distance statistic': input_details[i]['JS Distance statistic'],
                            'Threshold used in JS Distance': input_details[i]['Threshold used in JS Distance'],
                            'Drift status given by JS Distance': input_details[i]['Drift status given by JS Distance'],
                            'Final drift status by voting': input_details[i]['Final drift status']}
            
            data_quality[i] = {'Number of missing values': input_details[i]['Number of missing values'] if model_type != "Computer Vision (CV)" else 0,
                              'Uniqueness Score': input_details[i]['Uniqueness Score'] if model_type != "Computer Vision (CV)" else 100,
                              'Number of datatype mismatches': input_details[i]['Number of datatype mismatches'] if model_type != "Computer Vision (CV)" else 0,
                              'Number of outliers': input_details[i]['Number of outliers'] if model_type != "Computer Vision (CV)" else 0,
                               'Mode': input_details[i]['mode'],
                              'Count of unique categories': input_details[i]['value_counts'],
                               'Datatype': input_details[i]['type']
                              }
        
        
        data_drift['Drifting Features in Data'] = [i for i in cols if input_details[i]['Final drift status'] == True]

    
    if model_type == "Natural Language Processing (NLP)":
        data_drift['text'] =  input_details['text']
        data_quality['text'] = input_details['text']
    
    data_drift['Drift Detected using Copula'] = {"Top 3 Features' Relationship change": input_details["Top 3 Features' Relationship change"],
                                                     'Relationship Drift Detected': input_details['Relationship Drift Detected']} if 'Relationship Drift Detected' in input_details.keys() else {}
        
            
    
    # Prediction Dict
    prediction_drift = {"Output Feature Analysis": results['Output Feature Details']}
    
    # Production Data Details
    prod_summary = dict()

    for i in list(results['Production Data Summary'].keys()):
        if i not in ['Metrics', 'Drifting Metrics']:
            prod_summary[i] = results['Production Data Summary'][i]
    
    return prod_summary, performance_dict, data_drift, data_quality, prediction_drift           


def get_drift_dfs(col_type, ks_results=None, chi_results=None, psi_results=None, psi_num_results=None, js_results=None, js_num_results=None, num_ft=None, cat_ft=None):
    
    
    if col_type == "numerical":
        
        drift_num, drifting_nums = pd.DataFrame(), []
        
        for i in num_ft:
            status = [ks_results[i]['ks_status'], js_num_results[i]['js_status'], psi_num_results[i]['psi_status']]

            if status.count(True) > status.count(False):
                drifting_nums.append(i)
        ks_num = pd.DataFrame({'Feature Name': [i for i in drifting_nums],
                      'Test': ['KS-Test'] * len(drifting_nums),
                       'Statistic/Metric Value' : [ks_results[i]['ks-stat'] for i in drifting_nums],
                       'Threshold/Alpha': [0.05] * len(drifting_nums),
                      'P-value': [ks_results[i]['ks_p-value'] for i in drifting_nums],
                      'Test Status': [ks_results[i]['ks_status'] for i in drifting_nums]})
    
        psi_num = pd.DataFrame({'Feature Name': [i for i in drifting_nums],
                          'Test': ['PSI'] * len(drifting_nums),
                           'Statistic/Metric Value' : [psi_num_results[i]['psi_value'] for i in drifting_nums],
                          'Threshold/Alpha': [psi_num_results[i]['psi_threshold'] for i in drifting_nums],
                            'P-value': [np.nan] * len(drifting_nums),
                          'Test Status': [psi_num_results[i]['psi_status'] for i in drifting_nums]})

        js_num = pd.DataFrame({'Feature Name': [i for i in drifting_nums],
                          'Test': ['JS Divergence'] * len(drifting_nums),
                           'Statistic/Metric Value' : [js_num_results[i]['jsd_value'] for i in drifting_nums],
                          'Threshold/Alpha': [js_num_results[i]['js_threshold'] for i in drifting_nums],
                            'P-value': [np.nan] * len(drifting_nums),
                          'Test Status': [js_num_results[i]['js_status'] for i in drifting_nums]})

        drift_num = pd.concat([ks_num, psi_num, js_num], axis=0)
        
        return drift_num, drifting_nums
        
    if col_type == "categorical":
        
        drifting_cats, drift_cat = [], pd.DataFrame()
        
        for i in cat_ft:
            status = [chi_results[i]['chi_status'], js_results[i]['js_status'], psi_results[i]['psi_status']]

            if status.count(True) > status.count(False):
                drifting_cats.append(i)

        chi = pd.DataFrame({'Feature Name': [i for i in drifting_cats],
                          'Test': ['Chi-Square Test'] * len(drifting_cats),
                           'Statistic/Metric Value' : [chi_results[i]['chi'] for i in drifting_cats],
                           'Threshold/Alpha': [0.05] * len(drifting_cats),
                          'P-value': [chi_results[i]['chi_p-value'] for i in drifting_cats],
                          'Test Status': [chi_results[i]['chi_status'] for i in drifting_cats]})

        psi_cat = pd.DataFrame({'Feature Name': [i for i in drifting_cats],
                          'Test': ['PSI'] * len(drifting_cats),
                           'Statistic/Metric Value' : [psi_results[i]['psi_value'] for i in drifting_cats],
                          'Threshold/Alpha': [psi_results[i]['psi_threshold'] for i in drifting_cats],
                            'P-value': [np.nan] * len(drifting_cats),
                          'Test Status': [psi_results[i]['psi_status'] for i in drifting_cats]})

        js_cat = pd.DataFrame({'Feature Name': [i for i in drifting_cats],
                          'Test': ['JS Divergence'] * len(drifting_cats),
                           'Statistic/Metric Value' : [js_results[i]['jsd_value'] for i in drifting_cats],
                          'Threshold/Alpha': [js_results[i]['js_threshold'] for i in drifting_cats],
                            'P-value': [np.nan] * len(drifting_cats),
                          'Test Status': [js_results[i]['js_status'] for i in drifting_cats]})

        drift_cat = pd.concat([chi, psi_cat, js_cat], axis=0)
    
    return drift_cat, drifting_cats


def drift_using_copula(undrifted_data_df, drifted_data_df):
    
    DEGREE_OF_PRECISION = 30
    N_DRIFT_SUSPECTS = 3
    BASELINE_DATA_PORTION = 0.8
    CATEGORICAL_FEATURES, NUMERIC_FEATURES = determine_dtype_ft(undrifted_data_df)
    PYTORCH_SEED = 1001
    
    print(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    # cat_has_int, miss_dict, score, indices, outliers_index, num_invalid = validity_check(undrifted_data_df, drifted_data_df)

    categorizer = Categorizer(feature_names=CATEGORICAL_FEATURES) # Converts them into integers
    categorizer.fit(undrifted_data_df)
    
    undrifted_data_tensor = transfrom_unprocessed_df_to_pytorch(undrifted_data_df, categorizer, NUMERIC_FEATURES, CATEGORICAL_FEATURES) # Converts the df to pytorch tensor
    drifted_data_tensor = transform_file_to_pytorch(drifted_data_df, categorizer, NUMERIC_FEATURES, CATEGORICAL_FEATURES) # production run here
    
    n_undrifted_datapoints, n_features = undrifted_data_tensor.shape
    baseline_data_threshold = int(n_undrifted_datapoints * BASELINE_DATA_PORTION)
    torch.manual_seed(PYTORCH_SEED)
    permutation = torch.randperm(n_undrifted_datapoints)
    undrifted_data_tensor = undrifted_data_tensor[permutation]
    
    baseline_undrifted_data_tensor = undrifted_data_tensor[:baseline_data_threshold]
    test_undrifted_data_tensor = undrifted_data_tensor[baseline_data_threshold:]

    features_are_numeric = torch.concat((torch.full((len(NUMERIC_FEATURES),), True), torch.full((len(CATEGORICAL_FEATURES),), False)))

    baseline_copula = EmpiricalCopula(DEGREE_OF_PRECISION)
    baseline_copula.fit(baseline_undrifted_data_tensor, features_are_numeric=features_are_numeric)
    
    undrifted_copula = EmpiricalCopula(DEGREE_OF_PRECISION)
    undrifted_copula.fit(test_undrifted_data_tensor, features_are_numeric=features_are_numeric)
    
    potentially_drifted_copula = EmpiricalCopula(DEGREE_OF_PRECISION)
    potentially_drifted_copula.fit(drifted_data_tensor, features_are_numeric=features_are_numeric)

    undrifted_copula_distances = torch.zeros((n_features, n_features))
    for i in range(n_features):
      for j in range(i, n_features):
        baseline_copula_subset = baseline_copula.copula_subset(torch.tensor([i, j]))
        undrifted_copula_subset = undrifted_copula.copula_subset(torch.tensor([i, j]))
        current_distance = copula_distance(baseline_copula_subset, undrifted_copula_subset)
    
        undrifted_copula_distances[i, j] = current_distance
        undrifted_copula_distances[j, i] = current_distance

    drifted_copula_distances = torch.zeros((n_features, n_features))
    for i in range(n_features):
      for j in range(i, n_features):
        baseline_copula_subset = baseline_copula.copula_subset(torch.tensor([i, j]))
        drifted_copula_subset = potentially_drifted_copula.copula_subset(torch.tensor([i, j]))
        current_distance = copula_distance(baseline_copula_subset, drifted_copula_subset)
    
        drifted_copula_distances[i, j] = current_distance
        drifted_copula_distances[j, i] = current_distance

    distance_differences = drifted_copula_distances - undrifted_copula_distances

    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    fig = go.Figure(data=go.Heatmap(
                    z=distance_differences,
                    x=feature_names, 
                    y=feature_names))
    
    
    # List ordered by numerical first and categorical next
    relationship_indices = torch.triu_indices(n_features, n_features, offset=1).T
    flattened_distance_differences = torch.tensor([distance_differences[relationship_indices[i, 0], relationship_indices[i, 1]] for i in range(relationship_indices.shape[0])])
    maximum_distance_indices = torch.stack(torch.unravel_index(torch.topk(flattened_distance_differences, k=N_DRIFT_SUSPECTS)[1], (n_features, n_features)), dim=1)
    # print(maximum_distance_indices)

    feature_list_copula = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    feature_tensor = np.array(feature_list_copula)

    flattened_undrifted_distances = torch.tensor([undrifted_copula_distances[relationship_indices[i, 0], relationship_indices[i, 1]] for i in range(relationship_indices.shape[0])])
    drifted_threshold = torch.max(flattened_undrifted_distances)

    is_drift_detected = drifted_copula_distances >= drifted_threshold
    drift_detected_indices = torch.nonzero(is_drift_detected)
    drift_detected_indices = drift_detected_indices[drift_detected_indices[:, 0] < drift_detected_indices[:, 1]]
    
    seen = set()
    unique_pairs = []

    for pair in feature_tensor[drift_detected_indices].tolist():
        pair_set = frozenset(pair)
        if pair_set not in seen:
            seen.add(pair_set)
            unique_pairs.append(pair)


    return feature_tensor[maximum_distance_indices].tolist(), unique_pairs, fig


def get_results(model_name, date, ground_truth, production_data, baseline_data, model_type, benchmark_metrics, df=None):
    
    cols = production_data.drop(["target"],axis=1).columns # if model_type != "Computer Vision (CV)" else production_data.drop(["target", "paths"],axis=1).columns
    
    # figures = dict()
    
    categorical_ft, numerical_ft = determine_dtype_ft(baseline_data.drop(["target"],axis = 1))
    ks_results , ks_figs = ks_test(baseline_data.drop(["target"],axis = 1),production_data.drop("target",axis=1),num_ft = numerical_ft)
    chi_results, chi_figs = chi_test(baseline_data.drop(["target"],axis = 1),production_data.drop("target",axis=1),categorical_ft=categorical_ft)
    psi_results, psi_figs = psi(baseline_data.drop(["target"],axis = 1),production_data.drop("target",axis=1),categorical_ft=categorical_ft)
    js_results, js_figs = js_test(baseline_data.drop(["target"],axis = 1),production_data.drop("target",axis=1),features=categorical_ft)
    psi_num_results, psi_num_figs = psi_num(baseline_data.drop(["target"],axis = 1),production_data.drop("target",axis=1),num_ft = numerical_ft)
    js_num_results, js_num_figs = js_num(baseline_data.drop(["target"],axis = 1),production_data.drop("target",axis=1),num_ft = numerical_ft)
    
    # print(ks_results, chi_results, psi_results, js_results, psi_num_results, js_num_results)
    
    cat_has_int, miss_dict, score, indices, outliers_index, num_invalid = validity_check(baseline_data, production_data)
    
    # if (len(indices) == 0) or (score == 100):
    #     maximum_drift, drifted_features, fig = drift_using_copula(baseline_data.drop(["target"],axis=1), production_data.drop("target",axis=1))
    # else:
    #     # st.sidebar.warning("❗Invalid Data found and was removed for this analysis, Check Data Quality for more insights on this")
    #     maximum_drift, drifted_features, fig = drift_using_copula(baseline_data.drop(["target"],axis=1), production_data.drop(indices, axis=0).drop("target",axis=1))   
    
    
    true_labels, pred_labels = ground_truth['target'], production_data['target']
    true_labels = true_labels[true_labels.index.isin(pred_labels.index)]
    
    if model_type == "Classification" :
    
        gt, prod = EncodeLabels(true_labels, pred_labels)
        
        metrics = classification_metrics(gt, prod, list(baseline_data['target'].unique()))
        pred_results, pred_fig = chi_test(ground_truth['target'],production_data['target'], alpha = 0.05)
        pred_psi_results, pred_psi_figs = psi(ground_truth['target'],production_data['target'])
        pred_js_results, pred_js_figs = js_test(ground_truth['target'],production_data['target'])
        
        # figures['target'] = {'chi': pred_fig, 'psi': pred_psi_figs, 'js': pred_js_figs}
        
        # print(pred_js_results, pred_psi_results)
        
        pred_results.update(pred_psi_results)
        pred_results.update(pred_js_results)
        pred_stats = get_statistics(production_data['target'], "categorical", drift_results={'target': pred_results})
            
        performance = {
                "Metrics" : metrics ,  # check fpr 
                "Drifting Metrics" : [i for i in list(benchmark_metrics.keys()) if metrics[i] < benchmark_metrics[i]]
            }
    
    if model_type == "Regression":
        
        metrics = regression_metrics(true_labels, pred_labels)
        pred_results, pred_fig = ks_test(ground_truth['target'],production_data['target'], alpha = 0.05)
        pred_psi_results, pred_psi_figs = psi_num(ground_truth['target'],production_data['target'])
        pred_js_results, pred_js_figs = js_num(ground_truth['target'],production_data['target'])
        
        # figures['target'] = {'ks': pred_fig, 'psi': pred_psi_figs, 'js': pred_js_figs}
        
        pred_results.update(pred_psi_results)
        pred_results.update(pred_js_results)
        pred_stats = get_statistics(production_data['target'], "numerical", drift_results={'target': pred_results})
        
        performance = {
                "Metrics" : metrics ,  
                "Drifting Metrics" : [i for i in list(benchmark_metrics.keys()) if metrics[i] > benchmark_metrics[i]]
            }
    
    if model_type == "Natural Language Processing (NLP)":
        # pass
        categorical_ft = categorical_ft.remove('text')
        
        true_labels, pred_labels = EncodeLabels(ground_truth['target'], production_data['target'])
        metrics = classification_metrics(true_labels, pred_labels)
        metrics.pop("report")
        pred_results, pred_fig = chi_test(ground_truth['target'],production_data['target'], alpha = 0.05)
        pred_psi_results, pred_psi_figs = psi(ground_truth['target'],production_data['target'])
        pred_js_results, pred_js_figs = js_test(ground_truth['target'],production_data['target'])
        
        # figures['target'] = {'chi': pred_fig, 'psi': pred_psi_figs, 'js': pred_js_figs}
        performance = {
                "Metrics" : metrics ,  
                "Drifting Metrics" : [i for i in list(benchmark_metrics.keys()) if metrics[i] < benchmark_metrics[i]]
            }
        
        baseline_helper = pd.read_csv(f"./pages/models/{model_name}/helper/baseline.csv")
        production_helper = pd.read_csv(f"./pages/models/{model_name}/helper/{date}.csv")
        read_max_b, read_min_b, read_avg_b, max_text_read_b, min_text_read_b, _, _ = nlp_quality_metrics(baseline_helper, 'readability')
        read_max_p, read_min_p, read_avg_p, max_text_read_p, min_text_read_p, _, _ = nlp_quality_metrics(production_helper, 'readability')
        prod_words = get_words(production_data)
        number, spell_score, example = check_spelling(prod_words)
        length_max_b, length_min_b, length_avg_b, max_text_length_b, min_text_length_b, _, _ = nlp_quality_metrics(baseline_helper, 'length')
        length_max_p, length_min_p, length_avg_p, max_text_length_p, min_text_length_p, _, _ = nlp_quality_metrics(production_helper, 'length')
        words_b = get_words(baseline_data)
        syntax_figures, base_freq, prod_freq, uncommon, uncommon_freq, top5_uncommon, top5_b, top5_p = syntax_drift(words_b, prod_words)
        score = round((len(uncommon) / len(set(words_b))) * 100, 2)
        ttr = round(len(set(prod_words)) / len(prod_words), 2) * 100
        semantic_figure, isolated_score, isolated, edges, edge_alpha, pr_node_sizes, bs_node_sizes = semantic_drift(date, model_name)
        sorted_words_baseline = dict(sorted(base_freq.items(), key=lambda a:a[1], reverse=True))
        sorted_words_production = dict(sorted(prod_freq.items(), key=lambda a:a[1], reverse=True))
        freq_result, freq_fig = ks_test(list(sorted_words_baseline.values())[:1000], list(sorted_words_production.values())[:1000], nlp=True)
        freq_psi_result, freq_psi_fig = psi_num(pd.Series(list(sorted_words_baseline.values())[:1000]), pd.Series(list(sorted_words_production.values())[:1000]), nlp=True)
        freq_js_result, freq_js_fig = js_num(pd.Series(list(sorted_words_baseline.values())[:1000]), pd.Series(list(sorted_words_production.values())[:1000]), nlp=True)
        freq_result.update(freq_psi_result)
        freq_result.update(freq_js_result)
        status_list = [freq_result["ks_status"],freq_psi_result["psi_status"],freq_js_result["js_status"]]
        freq_result["Final Drift Status"] = True if status_list.count(True) >= 2 else False
        text_feature = {
            "Readability": {
                "Maximum" : read_max_p,
                "Minimum" : read_min_p,
                "Average" : read_avg_p,
                # "max_text_read_p" : max_text_read_p,
                # "min_text_read_p" : min_text_read_p
            },
            "Length" : {
                "Maximum" : length_max_p,
                "Minimum" : length_min_p,
                "Average" : length_avg_p,
                # "min_text_length_p" : min_text_length_p,
                # "max_text_length_p" : max_text_length_p
            },
            "Spelling error": round(spell_score,2),
            "Vocabulary" : len(set(prod_words)),
            "Misspelled words": number,
            "Correct words" : len(set(prod_words)) - number,
            "Syntax Drift": {
                "Type-Token Ratio" : {
                    "TTR Value(%)": ttr,
                    "Status": bool(ttr>=40 and ttr<=60)
                }  ,
                "Vocabulary drift" : {
                    "Score (%)": score,
                    "Status": bool(score>30)
                },
                "Frequency Based Syntax Drift" : freq_result
            },
            "Semantic Drift": {
                "Isolated Score": {
                    "Score": round(isolated_score,2),
                    "Status": bool(isolated_score>50)
                },
                "No.of Isolated Nodes": len(isolated)
            }
            
        }
        pred_results.update(pred_psi_results)
        pred_results.update(pred_js_results)
        pred_stats = get_statistics(production_data['target'], "categorical", drift_results={'target': pred_results})

    if model_type == "Text":
        categorical_ft = categorical_ft.remove('text')
        metrics = seq2seq_metrics(ground_truth['target'], production_data['target'])
        gt_helper = pd.read_csv(f'./pages/models/{model_name}/helper_target/Ground Truths/{date}.csv')
        prod_helper = pd.read_csv(f'./pages/models/{model_name}/helper_target/Production Runs/{date}.csv')
        metrics['Similarity'] = round(np.mean(prod_helper['similarity_with_gt']), 2)
        read_results, _ = ks_test(gt_helper['readability'], prod_helper['readability'], b_name="Ground Truth")
        read_psi_results, read_psi_figs = psi_num(gt_helper['readability'], prod_helper['readability'], b_name="Ground Truth")
        read_js_results, read_js_figs = js_num(gt_helper['readability'], prod_helper['readability'])
        read_max, read_min, read_avg, *_ = nlp_quality_metrics(prod_helper, 'readability', col='target')
        result_length, _ = ks_test(gt_helper['length'], prod_helper['length'], b_name="Ground Truth")
        psi_length , psi_length_fig = psi_num(gt_helper['length'], prod_helper['length'], b_name="Ground Truth")
        js_length , psi_length_fig = js_num(gt_helper['length'], prod_helper['length'])
        length_max, length_min, length_avg, *_ = nlp_quality_metrics(prod_helper, 'length', col='target')
        prod_words = get_words(prod_helper, col='target')
        number, spell_score, _ = check_spelling(prod_words)
        sim_results_gt, sim_fig_gt = ks_test(gt_helper['similarity'], prod_helper['similarity_wth_text'], b_name="Ground Truth")
        sim_psi_results_gt, sim_psi_fig_gt = psi_num(gt_helper['similarity'], prod_helper['similarity_wth_text'], b_name="Ground Truth")
        sim_js_results_gt, simp_js_fig_gt = js_num(gt_helper['similarity'], prod_helper['similarity_wth_text'])
        sim_ks_results_prod, _ = ks_test(prod_helper['similarity_with_gt'], prod_helper['similarity_wth_text'], b_name="Ground Truth")
        sim_psi_results_prod , sim_psi_fig_prod = psi_num(prod_helper['similarity_with_gt'], prod_helper['similarity_wth_text'], b_name="Ground Truth")
        sim_js_results_prod , sim_js_fig_prod = js_num(prod_helper['similarity_with_gt'], prod_helper['similarity_wth_text'])
        pred_results = {
            "readability": {"Maximum": read_max, "Minimum": read_min, "Average": read_avg},
            "KS-Test for Readability": read_results,
            "Length": {"Maximum": length_max, "Minimum": length_min, "Average": length_avg},
             "KS-Test for Length": result_length,
            "KS-Test for Similarity_gt": sim_results_gt ,
            "KS-Test for Similarity_prod": sim_ks_results_prod,
            "Spelling error": round(spell_score, 2),
            "vocabulary": len(set(prod_words)),
            "Misspelled words": number,
            "Correct words": len(set(prod_words)) - number,
        } 
        pred_psi_results = {
            "PSI for Readability" : read_psi_results,
            "PSI for Length" : psi_length,
            "PSI for Similarity_gt" : sim_psi_results_gt ,
            "PSI for Similarity_prod" : sim_psi_results_prod
            
        }
        pred_js_results  = {
            "JS for Readability" : read_js_results ,
            "JS for Length" : js_length ,
            "JS for Similarity_gt" : sim_js_results_gt,
            "JS for Similarity_prod" : sim_js_results_prod,
        }
        
        status_read_list = [read_results["ks_status"], read_psi_results["psi_status"], read_js_results["js_status"]]
        status_length_list = [result_length['ks_status'] , psi_length['psi_status'] , js_length['js_status']]
        status_sim_gt_list = [sim_results_gt['ks_status'] , sim_psi_results_gt['psi_status'] , sim_js_results_gt['js_status']]
        status_sim_prod_list = [sim_ks_results_prod['ks_status'] , sim_psi_results_prod['psi_status'] , sim_js_results_prod['js_status']]
        
        pred_final_results = {
            'Final drift status for Readability' : sum(bool(s) for s in status_read_list) >= 2,
            'Final drift status for length' : sum(bool(s) for s in status_length_list) >= 2,
            'Final drift status for similarity in GT' : sum(bool(s) for s in status_sim_gt_list) >= 2,
            'Final drift status for similarity in Prod' : sum(bool(s) for s in status_sim_prod_list) >= 2,
            
        }
        pred_js_results.update(pred_final_results)

        pred_results.update(pred_psi_results)
        pred_results.update(pred_js_results)
        pred_stats = pred_results
        
        performance = {
                "Metrics" : metrics ,  
                "Drifting Metrics" : [i for i in list(benchmark_metrics.keys()) if metrics[i] < benchmark_metrics[i]]
     }
        
        baseline_helper = pd.read_csv(f"./pages/models/{model_name}/helper/baseline.csv")
        production_helper = pd.read_csv(f"./pages/models/{model_name}/helper/{date}.csv")
        read_max_b, read_min_b, read_avg_b, max_text_read_b, min_text_read_b, _, _ = nlp_quality_metrics(baseline_helper, 'readability')
        read_max_p, read_min_p, read_avg_p, max_text_read_p, min_text_read_p, _, _ = nlp_quality_metrics(production_helper, 'readability')
        prod_words = get_words(production_data)
        number, spell_score, example = check_spelling(prod_words)
        length_max_b, length_min_b, length_avg_b, max_text_length_b, min_text_length_b, _, _ = nlp_quality_metrics(baseline_helper, 'length')
        length_max_p, length_min_p, length_avg_p, max_text_length_p, min_text_length_p, _, _ = nlp_quality_metrics(production_helper, 'length')
        words_b = get_words(baseline_data)
        syntax_figures, base_freq, prod_freq, uncommon, uncommon_freq, top5_uncommon, top5_b, top5_p = syntax_drift(words_b, prod_words)
        score = round((len(uncommon) / len(set(words_b))) * 100, 2)
        ttr = round(len(set(prod_words)) / len(prod_words), 2) * 100
        semantic_figure, isolated_score, isolated, edges, edge_alpha, pr_node_sizes, bs_node_sizes = semantic_drift(date, model_name)
        sorted_words_baseline = dict(sorted(base_freq.items(), key=lambda a:a[1], reverse=True))
        sorted_words_production = dict(sorted(prod_freq.items(), key=lambda a:a[1], reverse=True))
        freq_result, freq_fig = ks_test(list(sorted_words_baseline.values())[:1000], list(sorted_words_production.values())[:1000], nlp=True)
        freq_psi_result, freq_psi_fig = psi_num(pd.Series(list(sorted_words_baseline.values())[:1000]), pd.Series(list(sorted_words_production.values())[:1000]), nlp=True)
        freq_js_result, freq_js_fig = js_num(pd.Series(list(sorted_words_baseline.values())[:1000]), pd.Series(list(sorted_words_production.values())[:1000]), nlp=True)
        freq_result.update(freq_psi_result)
        freq_result.update(freq_js_result)
        status_list = [freq_result["ks_status"],freq_psi_result["psi_status"],freq_js_result["js_status"]]
        freq_result["Final Drift Status"] = True if status_list.count(True) >= 2 else False
        text_feature = {
            "Readability": {
                "Maximum" : read_max_p,
                "Minimum" : read_min_p,
                "Average" : read_avg_p,
                # "max_text_read_p" : max_text_read_p,
                # "min_text_read_p" : min_text_read_p
            },
            "Length" : {
                "Maximum" : length_max_p,
                "Minimum" : length_min_p,
                "Average" : length_avg_p,
                # "min_text_length_p" : min_text_length_p,
                # "max_text_length_p" : max_text_length_p
            },
            "Spelling error": round(spell_score,2),
            "Vocabulary" : len(set(prod_words)),
            "Misspelled words": number,
            "Correct words" : len(set(prod_words)) - number,
            "Syntax Drift": {
                "Type-Token Ratio" : {
                    "TTR Value(%)": ttr,
                    "Status": bool(ttr>=40 and ttr<=60)
                },
                "Vocabulary drift" : {
                    "Score (%)": score,
                    "Status": bool(score>30)
                },
                "Frequency Based Syntax Drift" : freq_result
            },
            "Semantic Drift": {
                "Isolated Score": {
                    "Score": round(isolated_score,2),
                    "Status": bool(isolated_score>50)
                },
                "No.of Isolated Nodes": len(isolated)
            }
            
        }
    
    if model_type == "Computer Vision (CV)":
        
        # Performance Drift
        gt, prod = EncodeLabels(ground_truth['target'],production_data['target'])
        metrics = classification_metrics(gt, prod, list(baseline_data['target'].unique()))
        
        # print(list(benchmark_metrics[model_name].keys()))
        
        performance = {
                "Metrics" : metrics ,  # check fpr 
                "Drifting Metrics" : [i for i in list(benchmark_metrics.keys()) if metrics[i] < benchmark_metrics[i]]
            }
        
        # baseline_data.drop('paths', axis=1, inplace=True)
        # production_data.drop('paths', axis=1, inplace=True)
        
        # Reading CSV with image information
        # df = pd.read_csv(f"../pages/models/{model_name}/Production/{date}/production_paths_df.csv")
        
        # Prediction Drift Analysis
        
        pred_results, pred_fig = chi_test(ground_truth['target'],production_data['target'], alpha = 0.05)
        pred_psi_results, pred_psi_figs = psi(ground_truth['target'],production_data['target'])
        pred_js_results, pred_js_figs = js_test(ground_truth['target'],production_data['target'])

        
        pred_results.update(pred_psi_results)
        pred_results.update(pred_js_results)
        pred_stats = get_statistics(production_data['target'], "categorical", drift_results={'target': pred_results})
        
        # pred_results, pred_fig = ks_test(ground_truth['target'],production_data['target'], alpha = 0.05)
        # pred_psi_results, pred_psi_figs = psi_num(ground_truth['target'],production_data['target'])
        # pred_js_results, pred_js_figs = js_num(ground_truth['target'],production_data['target'])
        
        # pred_results.update(pred_psi_results)
        # pred_results.update(pred_js_results)
        # pred_stats = get_statistics(production_data['target'], "categorical", drift_results={'target': pred_results})
        
        # figures['target'] = {'chi': pred_fig, 'psi': pred_psi_figs, 'js': pred_js_figs}


    if model_type != "Computer Vision (CV)":    
        
        completeness, _, WO_nan, w_nan, missing_data = data_completeness(production_data.drop(["target"],axis=1))
        uniqueness_score,u_s, _, _, no_of_rows_wo_dup = data_uniqueness(production_data.drop(["target"],axis=1))
        cat_mismatch, miss_dict, score,_, outliers_index, num_invalid = validity_check(baseline_data.drop("target",axis=1),production_data.drop(["target"],axis=1))
    
        
        summary = {
            "Number of rows" : len(production_data),
            "Number of rows without NaN" : float(WO_nan),
            "Number of rows with Nan" : float(w_nan),
            "Completeness Score" : float(completeness),
            "Uniqueness Score" : float(uniqueness_score) ,
            "Number of rows without duplicates" : float(no_of_rows_wo_dup),
            "Number of rows with duplicates" : float(len(production_data) - no_of_rows_wo_dup),
            

        }
        
        #     cols = production_data.drop(["probs","target"],axis=1).columns
        input_features = dict()
        input_u_s = {u_s['Feature'][i] : round(u_s['Value'][i], 2) for i in range(len(u_s['Feature']))}
        datatype_mismatch = {i : len(cat_mismatch[i]) for i in cat_mismatch}
        outliers = {i : len(outliers_index[i]) for i in outliers_index}


        for i in cols:
            input_features[i] = {
                    "Number of missing values" : float(dict(missing_data)[i]),
                    "Uniqueness Score" : float(input_u_s[i]),
                    "Number of datatype mismatches" : float(datatype_mismatch[i]) if i in list(datatype_mismatch.keys()) else  0,
                    "Number of outliers" : float(outliers[i]),
                    }
            
            if categorical_ft != None and i in categorical_ft:
                
                chi_results[i].update(psi_results[i])
                chi_results[i].update(js_results[i])
                
                # figures[i] = {'chi': chi_figs[i], 'psi': psi_figs[i], 'js': js_figs[i]}
                
                input_features[i].update(get_statistics(production_data[i], "categorical", drift_results=chi_results))
            
            if i in numerical_ft:
                
                ks_results[i].update(psi_num_results[i])
                ks_results[i].update(js_num_results[i])
                
                # figures[i] = {'ks': ks_figs[i], 'psi': psi_num_figs[i], 'js': js_num_figs[i]}
                
                input_features[i].update(get_statistics(production_data[i], "numerical", drift_results=ks_results))
        if model_type == "Natural Language Processing (NLP)" or model_type=="Text":
            input_features["text"].update(text_feature)
        else:
            summary["Validity Score"] = float(score),
            summary["Number of Invalid rows"] = float(num_invalid)
    
    else:
        summary = {
            "Number of Images" : len(df),
            "Number of Unique Resolutions and Sizes found" : len(df['resolution'].unique()),
            "Unique Resolutions found" : df['resolution'].unique().tolist(),
            "Unique Sizes found": df['size'].unique().tolist(),
            "Statistics of Sharpness of all Images": {"Maximum": float(round(df['sharpness'].max(), 2)),
                                                     "Minimum": float(round(df['sharpness'].min(), 2)),
                                                     "Mean": float(round(df['sharpness'].mean(), 2))},
            "Statistics of Brightness of all Images": {"Maximum": float(round(df['brightness'].max(), 2)),
                                                     "Minimum": float(round(df['brightness'].min(), 2)),
                                                     "Mean": float(round(df['brightness'].mean(), 2))},
            "Statistics of Noise of all Images": {"Maximum": float(round(df['noise'].max(), 2)),
                                                     "Minimum": float(round(df['noise'].min(), 2)),
                                                     "Mean": float(round(df['noise'].mean(), 2))},
            "Number of Anomalies": float(np.sum(df['Anomaly'] == True))
        }
        
        input_features = {i: [] for i in cols}
        
        for i in cols:
            
            if i in categorical_ft:
                
                chi_results[i].update(psi_results[i])
                chi_results[i].update(js_results[i])
                
                # figures[i] = {'chi': chi_figs[i], 'psi': psi_figs[i], 'js': js_figs[i]}
                
                input_features[i] = get_statistics(production_data[i], "categorical", drift_results=chi_results)
            
            if i in numerical_ft:
                
                ks_results[i].update(psi_num_results[i])
                ks_results[i].update(js_num_results[i])
                
                # figures[i] = {'ks': ks_figs[i], 'psi': psi_num_figs[i], 'js': js_num_figs[i]}
                
                input_features[i] = get_statistics(production_data[i], "numerical", drift_results=ks_results)

    if model_type not in ['Natural Language Processing (NLP)','Text']:
        if (len(indices) == 0) or (score == 100):
            maximum_drift, drifted_features, fig = drift_using_copula(baseline_data.drop(["target"],axis=1), production_data.drop("target",axis=1))
        else:
        # st.sidebar.warning("❗Invalid Data found and was removed for this analysis, Check Data Quality for more insights on this")
            maximum_drift, drifted_features, fig = drift_using_copula(baseline_data.drop(["target"],axis=1), production_data.drop(indices, axis=0).drop("target",axis=1))

        relationship_drift = {"Top 3 Features' Relationship change": maximum_drift,
                          "Relationship Drift Detected": drifted_features}
    
        input_features.update(relationship_drift)

    summary.update(performance)

    
    results_dict = {model_name : {str(date) : {"Production Data Summary": summary, "Input Feature Details": input_features, "Output Feature Details": pred_stats}}}
    
    print("Results dictionary created")
    
    return results_dict


# def get_fishbone(model_type, results_data, model_name, production_run_date, benchmark_metrics=None):
    
#     # results_data = results[model_name][production_run_date]
#     model_type_lower = model_type.lower()

#     if model_type_lower in ['natural language processing (nlp)', 'text']:
#         categories = {
#             'Prediction Drift': [],
#             'Data Drift': {
#                 'Syntax Drift': [],
#                 'Semantic Drift': []
#             },
#             'Data Quality': [],
#             'Performance Drift': []
#         }

#         input_details = results_data['Input Feature Details']
#         for feature, details in input_details.items():
#             if feature.lower() == "text":
#                 syntax = details.get('Syntax Drift', {})
#                 vocab_drift = syntax.get('Vocabulary drift score (%)', None)
#                 ttr = syntax.get('Type-Token Ratio (%)', None)

#                 if vocab_drift is not None:
#                     categories['Data Drift']['Syntax Drift'].append(("Vocabulary Drift", f"Vocabulary Drift: {vocab_drift}%"))
#                 if ttr is not None:
#                     categories['Data Drift']['Syntax Drift'].append(("Type-Token Ratio", f"TTR: {ttr}%"))

#                 semantic = details.get('Semantic Drift', {})
#                 isolated_score = semantic.get('Isolated Score', None)
#                 isolated_nodes = semantic.get('No.of Isolated Nodes', None)

#                 if isolated_score is not None:
#                     categories['Data Drift']['Semantic Drift'].append(("Isolated Score", f"Isolated Score: {isolated_score}"))
#                 if isolated_nodes is not None:
#                     categories['Data Drift']['Semantic Drift'].append(("Isolated Nodes", f"Isolated Nodes: {isolated_nodes}"))

#                 syntax_present = len(categories["Data Drift"]["Syntax Drift"]) > 0
#                 semantic_present = len(categories["Data Drift"]["Semantic Drift"]) > 0

#                 if not syntax_present and not semantic_present:
#                     summary_label = "no drift"
#                 elif syntax_present and semantic_present:
#                     summary_label = "syntax and semantic drift present"
#                 elif semantic_present:
#                     summary_label = "semantic drift present"
#                 else:
#                     summary_label = "syntax drift present"

#                 categories["Data Drift"]["Summary"] = [("Data Drift Summary", summary_label)]

#             spelling_error = details.get('Spelling error', 0)
#             misspelled_words = details.get('Misspelled words', 0)
#             num_missing = details.get('Number of missing values', 0)
#             num_outliers = details.get('Number of outliers', 0)
#             num_duplicates = details.get('num_of_duplicates', 0)
#             num_mismatches = details.get('Number of datatype mismatches', 0)

#             quality_issues = []
#             if spelling_error > 0:
#                 quality_issues.append(f"Spelling Error: {spelling_error}%")
#             if misspelled_words > 0:
#                 quality_issues.append(f"Misspelled Words: {misspelled_words}")
#             if num_missing > 0:
#                 quality_issues.append(f"Missing: {num_missing}")
#             if num_outliers > 0:
#                 quality_issues.append(f"Outliers: {num_outliers}")
#             if num_duplicates > 0:
#                 quality_issues.append(f"Duplicates: {num_duplicates}")
#             if num_mismatches > 0:
#                 quality_issues.append(f"Mismatches: {num_mismatches}")

#             if quality_issues:
#                 categories['Data Quality'].append((feature, ", ".join(quality_issues)))

#         if model_type_lower == 'text':
#             output_details = results_data.get('Output Feature Details', {})
#             if output_details.get('Final drift status for Readability', False):
#                 categories['Prediction Drift'].append(("Readability", "Drift detected in Readability distribution"))
#             if output_details.get('Final drift status for length', False):
#                 categories['Prediction Drift'].append(("Length", "Drift detected in Length distribution"))
#             if output_details.get('Final drift status for similarity in GT', False):
#                 categories['Prediction Drift'].append(("Similarity_GT", "Drift detected in Similarity with Ground Truth"))
#             if output_details.get('Final drift status for similarity in Prod', False):
#                 categories['Prediction Drift'].append(("Similarity_Prod", "Drift detected in Similarity of Production Output"))

#             spelling_error = output_details.get('Spelling error', 0)
#             misspelled_words = output_details.get('Misspelled words', 0)
#             if spelling_error > 0:
#                 categories['Prediction Drift'].append(("Spelling Error", f"Spelling Error in Output: {spelling_error}%"))
#             if misspelled_words > 0:
#                 categories['Prediction Drift'].append(("Misspelled Words", f"Misspelled Words in Output: {misspelled_words}"))

#     else:
#         from collections import defaultdict
#         categories = {
#             'Prediction Drift': [],
#             'Input Feature Analysis': defaultdict(list),
#             'Data Quality': [],
#             'Performance Drift': []
#         }

#         input_details = results_data['Input Feature Details']
#         input_keys = list(input_details.keys())[:-2]  # exclude last 2 keys
#         for feature in input_keys:
#             details = input_details[feature]

#             num_missing = details.get('Number of missing values', 0)
#             num_outliers = details.get('Number of outliers', 0)
#             num_duplicates = details.get('num_of_duplicates', 0)
#             num_mismatches = details.get('Number of datatype mismatches', 0)

#             quality_issues = []
#             if num_missing > 0:
#                 quality_issues.append(f"Missing: {num_missing}")
#             if num_outliers > 0:
#                 quality_issues.append(f"Outliers: {num_outliers}")
#             if num_duplicates > 0:
#                 quality_issues.append(f"Duplicates: {num_duplicates}")
#             if num_mismatches > 0:
#                 quality_issues.append(f"Mismatches: {num_mismatches}")

#             if quality_issues:
#                 categories['Data Quality'].append((feature, ", ".join(quality_issues)))

#             if isinstance(details, dict) and details.get('Final drift status', False):
#                 feature_type = details.get('type', '').lower()
#                 if 'Chi Square statistic' in details:
#                     hover_text = f"""<b>{feature} Drift Tests</b><br>Chi-Square: {details.get('Chi Square statistic')}, p = {details.get('Chi Square p-value')}, Drift: {details.get('Drift status given by Chi Square')}<br>PSI: {details.get('PSI statistic')}, Threshold: {details.get('Threshold used in PSI')}, Drift: {details.get('Drift status given by PSI')}<br>JS: {details.get('JS Distance statistic')}, Threshold: {details.get('Threshold used in JS Distance')}, Drift: {details.get('Drift status given by JS Distance')}"""
#                 else:
#                     hover_text = f"""<b>{feature} Drift Tests</b><br>KS: {details.get('KS-Test statistic')}, p = {details.get('KS-Test p-value')}, Drift: {details.get('Drift status given by KS-Test')}<br>PSI: {details.get('PSI statistic')}, Threshold: {details.get('Threshold used in PSI')}, Drift: {details.get('Drift status given by PSI')}<br>JS: {details.get('JS Distance statistic')}, Threshold: {details.get('Threshold used in JS Distance')}, Drift: {details.get('Drift status given by JS Distance')}"""
#                 categories['Input Feature Analysis']['Data Drift'].append((feature, hover_text))

#         seen_pairs = set()
#         rel_drift = input_details.get('Relationship Drift Detected', [])
#         for pair in rel_drift:
#             sorted_pair = tuple(sorted(pair))
#             if sorted_pair not in seen_pairs:
#                 seen_pairs.add(sorted_pair)
#                 hover_text = f"Pairwise Drift: {sorted_pair[0]} ↔️ {sorted_pair[1]}"
#                 categories['Input Feature Analysis']['Copula'].append((f"{sorted_pair[0]}-{sorted_pair[1]}", hover_text))

#         data_drift_present = len(categories['Input Feature Analysis']['Data Drift']) > 0
#         copula_drift_present = len(categories['Input Feature Analysis']['Copula']) > 0
#         if not data_drift_present and not copula_drift_present:
#             summary_label = "no drift"
#         elif data_drift_present and copula_drift_present:
#             summary_label = "data and copula drift present"
#         elif data_drift_present:
#             summary_label = "data drift present"
#         else:
#             summary_label = "copula drift present"
#         categories['Input Feature Analysis']['Summary'] = [("Input Drift Summary", summary_label)]

#         if model_type_lower == "computer vision (cv)":
#             summary = results_data['Production Data Summary']
#             num_anomalies = summary.get('Number of Anomalies', 0)
#             if num_anomalies > 0:
#                 categories['Data Quality'].append(("Anomalies", f"{num_anomalies} anomalies detected"))
#             unique_res_sizes = summary.get('Number of Unique Resolutions and Sizes found', 0)
#             if unique_res_sizes > 0:
#                 categories['Data Quality'].append(("Image Variance", f"{unique_res_sizes} unique resolutions and sizes found"))

#     output_drift = results_data.get('Output Feature Details', {})
#     if output_drift.get('Final drift status', False):
#         if model_type_lower == 'regression':
#             hover_text = f"""<b>Prediction Drift (Numerical)</b><br>
# KS: {output_drift.get('KS-Test statistic')}, p = {output_drift.get('KS-Test p-value')}, Drift: {output_drift.get('Drift status given by KS-Test')}<br>
# PSI: {output_drift.get('PSI statistic')}, Threshold: {output_drift.get('Threshold used in PSI')}, Drift: {output_drift.get('Drift status given by PSI')}<br>
# JS: {output_drift.get('JS Distance statistic')}, Threshold: {output_drift.get('Threshold used in JS Distance')}, Drift: {output_drift.get('Drift status given by JS Distance')}"""
#         else:
#             hover_text = f"""<b>Prediction Drift (Categorical)</b><br>
# Chi-Square: {output_drift.get('Chi Square statistic')}, p = {output_drift.get('Chi Square p-value')}, Drift: {output_drift.get('Drift status given by Chi Square')}<br>
# PSI: {output_drift.get('PSI statistic')}, Threshold: {output_drift.get('Threshold used in PSI')}, Drift: {output_drift.get('Drift status given by PSI')}<br>
# JS: {output_drift.get('JS Distance statistic')}, Threshold: {output_drift.get('Threshold used in JS Distance')}, Drift: {output_drift.get('Drift status given by JS Distance')}"""
#         categories['Prediction Drift'].append(("Target", hover_text))

#     metrics = results_data['Production Data Summary']['Metrics']
#     metric_abbr = {
#         "Accuracy": "Accuracy",
#         "F1": "F1",
#         "F1 Score": "F1",
#         "Precision": "Precision",
#         "Recall": "Recall",
#         "ROC_AUC": "ROC-AUC",
#         "Log Loss": "Log",
#         "False Positive Rate": "FPR",
#         "True Positive Rate": "TPR",
#         "Mean Absolute Error": "MAE",
#         "Mean Absolute Percentage Error": "MAPE",
#         "Mean Squared Error": "MSE",
#         "Root Mean Squared Error": "RMSE",
#         "R2 Score": "R²",
#         "BLEU": "BLEU",
#         "Similarity": "Similarity"
#     }
#     for metric, value in metrics.items():
#         if benchmark_metrics is None or metric not in benchmark_metrics:
#             continue
#         benchmark = benchmark_metrics[metric]
#         delta = value - benchmark if metric == ["False Positive Rate"] else benchmark - value
#         delta = (value - benchmark)*100/benchmark if metric in ["Mean Absolute Error", "Mean Absolute Percentage Error", "Mean Squared Error","Root Mean Squared Error"] else benchmark - value
#         show_change = (
#             (metric in ["False Positive Rate", "Mean Absolute Error", "Mean Absolute Percentage Error", "Mean Squared Error","Root Mean Squared Error"] and value > benchmark) or
#             (metric not in ["False Positive Rate", "Mean Absolute Error", "Mean Absolute Percentage Error", "Mean Squared Error","Root Mean Squared Error"] and value < benchmark)
#         )
#         if show_change:
#             metric_label = metric_abbr.get(metric, metric)
#             if metric in ["False Positive Rate", "Mean Absolute Error", "Mean Absolute Percentage Error", "Mean Squared Error","Root Mean Squared Error"] and value > benchmark:
#                 hover_text = f"{metric} increased by {delta:.2f}%"
#             else:
#                 hover_text = f"{metric} dropped by {delta:.2f}%"
#             categories['Performance Drift'].append((metric_label, hover_text))   


#     fig = go.Figure()
#     num_causes = len([k for k in categories.keys()])
#     base_length = 8
#     needs_subbone = model_type_lower in ["classification", "regression"] and "Input Feature Analysis" in categories
#     spine_padding = 0.15 if needs_subbone else 0

#     main_arrow_end_x = base_length / 2 + spine_padding
#     main_arrow_start_x = -base_length / 2 - spine_padding
#     main_arrow_y = 0

#     fig.add_annotation(
#         x=main_arrow_end_x, y=main_arrow_y,
#         ax=main_arrow_start_x - 11.5, ay=main_arrow_y,
#         xref="x", yref="y", axref="x", ayref="y",
#         showarrow=True,
#         arrowhead=3,
#         arrowsize=1.5,
#         arrowwidth=4,
#         arrowcolor="blue"
#     )

#     fig.add_annotation(
#         x=main_arrow_end_x + 1.2, y=main_arrow_y, 
#         xref="x", yref="y",
#         text="Model Drift",
#         showarrow=False,
#         font=dict(color="white", size=18),
#         bgcolor="purple",
#         bordercolor="black",
#         borderwidth=1,
#         borderpad=10,
#         align="center"
#     )

#     causes = list(categories.keys())
#     spacing = (main_arrow_end_x - main_arrow_start_x) / (len(causes) + 0.5)
#     cause_length = 10

#     for i, cause in enumerate(causes):
#         x_base = main_arrow_start_x + spacing * (i + 1)
#         y_base = main_arrow_y
#         is_top = (i % 2 == 0)
#         dy = cause_length * 2 if is_top else -cause_length * 2
#         dx = -cause_length * 0.7071
#         x_tip = x_base + dx
#         y_tip = y_base + dy

#         fig.add_annotation(
#             x=x_base, y=y_base,
#             ax=x_tip , ay=y_tip,
#             xref="x", yref="y", axref="x", ayref="y",
#             showarrow=True,
#             arrowhead=3,
#             arrowsize=1.5,
#             arrowwidth=2,
#             arrowcolor="green"
#         )

#         fig.add_annotation(
#             x=x_tip - 0.1, y=y_tip,
#             xref="x", yref="y",
#             text=cause,
#             showarrow=False,
#             font=dict(color="white" , size = 14.25),
#             bgcolor="green",
#             bordercolor="black",
#             borderwidth=1,
#             borderpad=4,
#             align="center"
#         )

#         subcauses = categories[cause]
        
# #         print(subcauses)

#         if isinstance(subcauses, dict):  # Nested structure
#             sub_spine_start_x = x_tip - 0.9
#             sub_spine_y = y_tip
#             sub_spine_end_x = sub_spine_start_x - 6

#             fig.add_annotation(
#                 ax=sub_spine_end_x, ay=sub_spine_y,
#                 x=sub_spine_start_x if cause == "Data Drift" else sub_spine_start_x - 1, 
#                 y=sub_spine_y,
#                 xref="x", yref="y", axref="x", ayref="y",
#                 showarrow=True,
#                 arrowhead=3,
#                 arrowsize=1.5,
#                 arrowwidth=3,
#                 arrowcolor="blue"
#             )
#             summary = subcauses.get("Summary", [])
#             other_subs = [(k, v) for k, v in subcauses.items() if k != "Summary"]

#             for j, (subtype, drift_causes) in enumerate(other_subs):
# #                 print(subtype)
#                 y_offset = 9 if subtype == 'Syntax Drift' or subtype=="Data Drift" else -9
                
#                 sub_x = sub_spine_end_x
#                 sub_y = sub_spine_y + y_offset

#                 fig.add_annotation(
#                     x=sub_spine_start_x - 2 if subtype == 'Syntax Drift' or subtype=="Data Drift" else sub_spine_start_x - 1.8, 
#                     y=sub_spine_y,
#                     ax=sub_x, ay=sub_y,
#                     xref="x", yref="y", axref="x", ayref="y",
#                     showarrow=True,
#                     arrowhead=3,
#                     arrowsize=1.5,
#                     arrowwidth=2,
#                     arrowcolor="green"
#                 )

#                 fig.add_annotation(
#                     x=sub_x - 0.2, y=sub_y,
#                     xref="x", yref="y",
#                     text=subtype,
#                     showarrow=False,
#                     font=dict(color="white" , size = 14),
#                     bgcolor="green",
#                     bordercolor="black",
#                     borderwidth=1,
#                     borderpad=4,
#                     align="center"
#                 )

#                 if len(drift_causes) == 0:
#                     t = 0.5
#                     base_x = sub_x + (sub_spine_start_x - 2 - sub_x) * t
#                     base_y = sub_y + (sub_spine_y - sub_y) * t
#                     sub_line_length = 3
#                     fig.add_shape(
#                         type="line",
#                         x0=base_x, y0=base_y,
#                         x1=base_x - sub_line_length, y1=base_y,
#                         line=dict(color="grey", width=1.5, dash="solid")
#                     )
#                     fig.add_trace(go.Scatter(
#                         x=[base_x - sub_line_length / 2],
#                         y=[base_y + 0.35],
#                         mode="text",
#                         text=["No Drift"],
#                         textfont=dict(color="black", size=13),
#                         showlegend=False,
#                         hoverinfo='skip'
#                     ))
#                 else:
#                     for k, (short_text, hover_text) in enumerate(drift_causes):
#                         t = (k + 1) / (len(drift_causes) + 1)
#                         base_x = sub_x + (sub_spine_start_x - 2 - sub_x) * t
#                         base_y = sub_y + (sub_spine_y - sub_y) * t
#                         sub_line_length = min(max(len(short_text) * 0.15, 3), 8)
#                         fig.add_shape(
#                             type="line",
#                             x0=base_x, y0=base_y,
#                             x1=base_x - sub_line_length, y1=base_y,
#                             line=dict(color="grey", width=1.5)
#                         )
#                         fig.add_trace(go.Scatter(
#                             x=[base_x - sub_line_length / 2],
#                             y=[base_y + 0.35],
#                             mode="text",
#                             text=[short_text],
#                             hovertext=[hover_text],
#                             hoverinfo="text",
#                             textfont=dict(color="black", size=11.75),
#                             showlegend=False
#                         ))
                        
                        
#             if summary:
#                 short_text, hover_text = summary[0]  # Only one summary item
#                 x_on_arrow = x_tip + (x_base - x_tip) * 0.5
#                 y_on_arrow = y_tip + (y_base - y_tip) * 0.5
#                 sub_line_length = min(max(len(hover_text) * 0.15, 3), 8)

#                 fig.add_shape(
#                         type="line",
#                         x0=x_on_arrow, y0=y_on_arrow,
#                         x1=x_on_arrow - sub_line_length, y1=y_on_arrow,
#                         line=dict(color="grey", width=1.5)
#                     )
#                 fig.add_trace(go.Scatter(
#                         x=[x_on_arrow - sub_line_length / 2],
#                         y=[y_on_arrow + 0.35],
#                         mode="text",
#                         text=[hover_text],
#                         hoverinfo="text",
#                         textfont=dict(color="black", size=13),
#                         showlegend=False
#                     ))
#             # Draw summary subcause under the main Input Feature Analysis arrow
            

#         elif model_type_lower == "natural language processing (nlp)" and cause == "Data Quality":
#             input_feature_details = results_data.get("Input Feature Details", {})
#             text_metrics = input_feature_details.get("text", {})

#             hover_lines = []
#             if "Spelling error" in text_metrics:
#                 hover_lines.append(f"Spelling Error: {text_metrics['Spelling error']}%")
#             if "Misspelled words" in text_metrics:
#                 hover_lines.append(f"Misspelled Words: {text_metrics['Misspelled words']}")
#             if "Number of outliers" in text_metrics:
#                 hover_lines.append(f"Outliers: {text_metrics['Number of outliers']}")
#             if "Number of mismatches" in text_metrics:
#                 hover_lines.append(f"Mismatches: {text_metrics['Number of mismatches']}")
#             if "Number of missing values" in text_metrics:
#                 hover_lines.append(f"Missing Values: {text_metrics['Number of missing values']}")
#             if "num_of_duplicates" in text_metrics:
#                 hover_lines.append(f"Duplicates: {text_metrics['num_of_duplicates']}")

#             combined_hover = "<br>".join(hover_lines) if hover_lines else "No Data Quality Issues"

#             x_on_arrow = x_tip + (x_base - x_tip) * 0.5
#             y_on_arrow = y_tip + (y_base - y_tip) * 0.5
#             sub_line_length = 3

#             fig.add_shape(
#                 type="line",
#                 x0=x_on_arrow, y0=y_on_arrow,
#                 x1=x_on_arrow - sub_line_length, y1=y_on_arrow,
#                 line=dict(color="grey", width=1.5)
#             )
#             fig.add_trace(go.Scatter(
#                 x=[x_on_arrow - sub_line_length / 2],
#                 y=[y_on_arrow + 0.35],
#                 mode="text",
#                 text=["text"],
#                 hovertext=[combined_hover],
#                 hoverinfo="text",
#                 textfont=dict(color="black", size=13.25),
#                 showlegend=False
#             ))
#         else:
#             if len(subcauses) == 0:
#                 x_on_arrow = x_tip + (x_base - x_tip) * 0.5
#                 y_on_arrow = y_tip + (y_base - y_tip) * 0.5
#                 sub_line_length = 3
#                 fig.add_shape(
#                     type="line",
#                     x0=x_on_arrow, y0=y_on_arrow,
#                     x1=x_on_arrow - sub_line_length, y1=y_on_arrow,
#                     line=dict(color="grey", width=1.5, dash="solid")
#                 )
#                 fig.add_trace(go.Scatter(
#                     x=[x_on_arrow - sub_line_length / 2],
#                     y=[y_on_arrow + 0.35],
#                     mode="text",
#                     text=["No Drift"],
#                     textfont=dict(color="black", size=13.25),
#                     showlegend=False,
#                     hoverinfo='text'
#                 ))
#             else:
#                 for j, subcause in enumerate(subcauses):
#                     frac = (j + 1) / (len(subcauses) + 0.5)
#                     x_on_arrow = x_tip + (x_base - x_tip) * frac
#                     y_on_arrow = y_tip + (y_base - y_tip) * frac
#                     if isinstance(subcause, tuple):
#                         short_text, hover_text = subcause
#                     else:
#                         short_text = subcause.split('\n')[0]
#                         hover_text = subcause
#                     sub_line_length = min(max(len(short_text) * 0.15, 3), 8)
#                     fig.add_shape(
#                         type="line",
#                         x0=x_on_arrow, y0=y_on_arrow,
#                         x1=x_on_arrow - sub_line_length, y1=y_on_arrow,
#                         line=dict(color="grey", width=1.5)
#                     )
#                     fig.add_trace(go.Scatter(
#                         x=[x_on_arrow - sub_line_length / 2],
#                         y=[y_on_arrow + 0.35],
#                         mode="text",
#                         text=[short_text],
#                         hovertext=[hover_text],
#                         hoverinfo="text",
#                         textfont=dict(color="black", size=13.25),
#                         showlegend=False
#                     ))
                    

#     fig.update_layout(
#         title=f"{model_name} - {model_type} Model Drift<br><sup>Production Run: {production_run_date}</sup>",
#         xaxis=dict(range=[-20, 8], showgrid=False, visible=False),
#         yaxis=dict(range=[-30, 23], showgrid=False, visible=False),
#         width=1400,
#         height=850
#     )
    
#     return fig

def fishbone_plot(model_type, results_data, model_name, production_run_date, benchmark_metrics=None):
    
    #results_data = results[model_name][production_run_date]
    model_type_lower = model_type.lower()

    if model_type_lower in ['natural language processing (nlp)', 'text']:
        categories = {
            'Prediction Drift': [],
            'Data Drift': {
                'Syntax Drift': [],
                'Semantic Drift': []
            },
            'Data Quality': [],
            'Performance Drift': []
        }

        input_details = results_data['Input Feature Details']
        for feature, details in input_details.items():
            # print(feature,details)
            if feature.lower() == "text":
                syntax = details.get('Syntax Drift', {})
                vocab_drift = syntax.get('Vocabulary drift', None)
                # print(vocab_drift)
                ttr = syntax.get('Type-Token Ratio', None)
                # print(ttr)
                freq_drift = syntax.get('Frequency Based Syntax Drift',None)

                if (vocab_drift is not None) and (vocab_drift["Status"]):
                    categories['Data Drift']['Syntax Drift'].append(("Vocabulary Drift", f"Vocabulary Drift: {vocab_drift['Score (%)']}%",f"rgb(255,{(255-(((vocab_drift['Score (%)']-30)/(100-30))*255))},0)"))
                if (ttr is not None) and (ttr["Status"]):
                    categories['Data Drift']['Syntax Drift'].append(("Type-Token Ratio", f"TTR: {ttr['TTR Value(%)']}%",f"rgb(255,{(255-(((ttr['TTR Value(%)']-40)/(60-40))*255))},0)"))
                if (freq_drift is not None) and (freq_drift["Final Drift Status"]):
                    bools = [freq_drift.get('ks_status'), freq_drift.get('psi_status'), freq_drift.get('js_status')]
                    color = "rgb(255, 0, 0)" if sum(bools) == 3 else "rgb(255, 165, 0)"
                    categories['Data Drift']['Syntax Drift'].append(("Frequency Based Syntax Drift", f"""KS-test: {freq_drift.get('ks-stat')}, p = {freq_drift.get('ks_p-value')}, Drift: {freq_drift.get('ks_status')}<br>PSI: {freq_drift.get('psi_value')}, Threshold: {freq_drift.get('psi_threshold')}, Drift: {freq_drift.get('psi_status')}<br>JS: {freq_drift.get('jsd_value')}, Threshold: {freq_drift.get('js_threshold')}, Drift: {freq_drift.get('js_status')}""", color))

                semantic = details.get('Semantic Drift', {})
                isolated_score = semantic.get('Isolated Score', None)
                isolated_nodes = semantic.get('No.of Isolated Nodes', None)
                # print(isolated_score)

                if (isolated_score is not None) and (isolated_score["Status"]):
                    categories['Data Drift']['Semantic Drift'].append(("Isolated Score", f"Isolated Score: {isolated_score['Score']},Isolated Nodes: {isolated_nodes}",f"rgb(255,{(255-(((isolated_score['Score']-50)/(100-50))*255))},0)"))
                
                syntax_present = len(categories["Data Drift"]["Syntax Drift"]) > 0
                semantic_present = len(categories["Data Drift"]["Semantic Drift"]) > 0

                if not syntax_present and not semantic_present:
                    summary_label = "no drift"
                    color = "green"
                elif syntax_present and semantic_present:
                    summary_label = "syntax and semantic drift present"
                    color = "rgb(255, 0, 0)"
                elif semantic_present:
                    summary_label = "semantic drift present"
                    color = "rgb(255, 165, 0)"
                else:
                    summary_label = "syntax drift present"
                    color = "rgb(255, 165, 0)"

                categories["Data Drift"]["Summary"] = [("Data Drift Summary", summary_label,color)]
            
            if feature not in ["Top 3 Features' Relationship change",'Relationship Drift Detected']:
                spelling_error = details.get('Spelling error', 0)
                misspelled_words = details.get('Misspelled words', 0)
                num_missing = details.get('Number of missing values', 0)
                num_outliers = details.get('Number of outliers', 0)
                num_duplicates = details.get('num_of_duplicates', 0)
                num_mismatches = details.get('Number of datatype mismatches', 0)
    
                quality_issues = []
                colors = []
                if spelling_error > 0:
                    quality_issues.append(f"Spelling Error: {spelling_error}%")
                    colors.append(f"rgb(255,{round(255-(((spelling_error-1)/(100-1))*255),2)},0)")
                if misspelled_words > 0:
                    quality_issues.append(f"Misspelled Words: {misspelled_words}")
                    colors.append(f"rgb(255,{round(255-(((misspelled_words-1)/(100-1))*255),2)},0)")
                if num_missing > 0:
                    quality_issues.append(f"Missing: {num_missing}")
                    colors.append(f"rgb(255,{round(255-(((num_missing-1)/(100-1))*255),2)},0)")
                if num_outliers > 0:
                    quality_issues.append(f"Outliers: {num_outliers}")
                    colors.append(f"rgb(255,{round(255-(((num_outliers-1)/(100-1))*255),2)},0)")
                if num_duplicates > 0:
                    quality_issues.append(f"Duplicates: {num_duplicates}")
                    colors.append(f"rgb(255,{round(255-(((num_duplicates-1)/(100-1))*255),2)},0)")
                if num_mismatches > 0:
                    quality_issues.append(f"Mismatches: {num_mismatches}")
                    colors.append(f"rgb(255,{round(255-(((num_mismatches-1)/(100-1))*255),2)},0)")

                if colors:
                    color = min(colors, key=lambda c: c[1])
                else:
                    color = "green"
                if quality_issues:
                    categories['Data Quality'].append((feature, ", ".join(quality_issues), color))

        if model_type_lower == 'text':
            output_details = results_data.get('Output Feature Details', {})
            if output_details.get('Final drift status for Readability', False):
                bools = [output_details.get('KS-Test for Readability').get('ks_status'), output_details.get("PSI for Readability").get('psi_status'), output_details.get("JS for Readability").get('js_status')]
                color = "rgb(255, 0, 0)" if sum(bools) == 3 else "rgb(255, 165, 0)"
                categories['Prediction Drift'].append(("Readability", f"""KS-test: {output_details.get('KS-Test for Readability').get('ks-stat')}, p = {output_details.get('KS-Test for Readability').get('ks_p-value')}, Drift: {output_details.get('KS-Test for Readability').get('ks_status')}<br>PSI: {output_details.get('PSI for Readability').get('psi_value')}, Threshold: {output_details.get('PSI for Readability').get('psi_threshold')}, Drift: {output_details.get('PSI for Readability').get('psi_status')}<br>JS: {output_details.get('JS for Readability').get('jsd_value')}, Threshold: {output_details.get('JS for Readability').get('js_threshold')}, Drift: {output_details.get('JS for Readability').get('js_status')}""", color))
            if output_details.get('Final drift status for length', False):
                bools = [output_details.get('KS-Test for Length').get('ks_status'), output_details.get("PSI for Length").get('psi_status'), output_details.get("JS for Length").get('js_status')]
                color = "rgb(255, 0, 0)" if sum(bools) == 3 else "rgb(255, 165, 0)"
                categories['Prediction Drift'].append(("Length", f"""KS-test: {output_details.get('KS-Test for Length').get('ks-stat')}, p = {output_details.get('KS-Test for Length').get('ks_p-value')}, Drift: {output_details.get('KS-Test for Length').get('ks_status')}<br>PSI: {output_details.get('PSI for Length').get('psi_value')}, Threshold: {output_details.get('PSI for Length').get('psi_threshold')}, Drift: {output_details.get('PSI for Length').get('psi_status')}<br>JS: {output_details.get('JS for Length').get('jsd_value')}, Threshold: {output_details.get('JS for Length').get('js_threshold')}, Drift: {output_details.get('JS for Length').get('js_status')}""", color))
            if output_details.get('Final drift status for similarity in GT', False):
                bools = [output_details.get('KS-Test for Similarity_gt').get('ks_status'), output_details.get("PSI for Similarity_gt").get('psi_status'), output_details.get("JS for Similarity_gt").get('js_status')]
                color = "rgb(255, 0, 0)" if sum(bools) == 3 else "rgb(255, 165, 0)"
                categories['Prediction Drift'].append(("Similarity_GT", f"""KS-test: {output_details.get('KS-Test for Similarity_gt').get('ks-stat')}, p = {output_details.get('KS-Test for Similarity_gt').get('ks_p-value')}, Drift: {output_details.get('KS-Test for Similarity_gt').get('ks_status')}<br>PSI: {output_details.get('PSI for Similarity_gt').get('psi_value')}, Threshold: {output_details.get('PSI for Similarity_gt').get('psi_threshold')}, Drift: {output_details.get('PSI for Similarity_gt').get('psi_status')}<br>JS: {output_details.get('JS for Similarity_gt').get('jsd_value')}, Threshold: {output_details.get('JS for Similarity_gt').get('js_threshold')}, Drift: {output_details.get('JS for Similarity_gt').get('js_status')}""", color))
            if output_details.get('Final drift status for similarity in Prod', False):
                bools = [output_details.get('KS-Test for Similarity_prod').get('ks_status'), output_details.get("PSI for Similarity_prod").get('psi_status'), output_details.get("JS for Similarity_prod").get('js_status')]
                color = "rgb(255, 0, 0)" if sum(bools) == 3 else "rgb(255, 165, 0)"
                categories['Prediction Drift'].append(("Similarity_Prod", f"""KS-test: {output_details.get('KS-Test for Similarity_prod').get('ks-stat')}, p = {output_details.get('KS-Test for Similarity_prod').get('ks_p-value')}, Drift: {output_details.get('KS-Test for Similarity_prod').get('ks_status')}<br>PSI: {output_details.get('PSI for Similarity_prod').get('psi_value')}, Threshold: {output_details.get('PSI for Similarity_prod').get('psi_threshold')}, Drift: {output_details.get('PSI for Similarity_prod').get('psi_status')}<br>JS: {output_details.get('JS for Similarity_prod').get('jsd_value')}, Threshold: {output_details.get('JS for Similarity_prod').get('js_threshold')}, Drift: {output_details.get('JS for Similarity_prod').get('js_status')}""", color))

            spelling_error = output_details.get('Spelling error', 0)
            misspelled_words = output_details.get('Misspelled words', 0)
            if spelling_error > 0:
                categories['Prediction Drift'].append(("Spelling Error", f"Spelling Error in Output: {spelling_error}%", f"rgb(255,{(255-(((spelling_error-1)/(100-1))*255))},0)"))
            if misspelled_words > 0:
                categories['Prediction Drift'].append(("Misspelled Words", f"Misspelled Words in Output: {misspelled_words}", f"rgb(255,{(255-(((misspelled_words-1)/(1000-1))*255))},0)"))

    else:
        from collections import defaultdict
        categories = {
            'Prediction Drift': [],
            'Input Feature Analysis': defaultdict(list),
            'Data Quality': [],
            'Performance Drift': []
        }

        input_details = results_data['Input Feature Details']
        input_keys = list(input_details.keys())[:-2]  # exclude last 2 keys
        for feature in input_keys:
            details = input_details[feature]

            num_missing = details.get('Number of missing values', 0)
            num_outliers = details.get('Number of outliers', 0)
            num_duplicates = details.get('num_of_duplicates', 0)
            num_mismatches = details.get('Number of datatype mismatches', 0)

            quality_issues = []
            colors = []
            if num_missing > 0:
                quality_issues.append(f"Missing: {num_missing}")
                colors.append(f"rgb(255,{round(255-(((num_missing-1)/(100-1))*255),2)},0)")
            if num_outliers > 0:
                quality_issues.append(f"Outliers: {num_outliers}")
                colors.append(f"rgb(255,{round(255-(((num_outliers-1)/(100-1))*255),2)},0)")
            if num_duplicates > 0:
                quality_issues.append(f"Duplicates: {num_duplicates}")
                colors.append(f"rgb(255,{round(255-(((num_duplicates-1)/(100-1))*255),2)},0)")
            if num_mismatches > 0:
                quality_issues.append(f"Mismatches: {num_mismatches}")
                colors.append(f"rgb(255,{round(255-(((num_mismatches-1)/(100-1))*255),2)},0)")
            
            if colors:
                color = min(colors, key=lambda c: c[1])
            else:
                color = "green"

            if quality_issues:
                categories['Data Quality'].append((feature, ", ".join(quality_issues), color))

            if isinstance(details, dict) and details.get('Final drift status', False):
                feature_type = details.get('type', '').lower()
                if 'Chi Square statistic' in details:
                    bools = [details.get('Chi Square statistic'), details.get('PSI statistic'), details.get('JS Distance statistic')]
                    color = "rgb(255, 0, 0)" if sum(bools) == 3 else "rgb(255, 165, 0)"
                    hover_text = f"""<b>{feature} Drift Tests</b><br>Chi-Square: {details.get('Chi Square statistic')}, p = {details.get('Chi Square p-value')}, Drift: {details.get('Drift status given by Chi Square')}<br>PSI: {details.get('PSI statistic')}, Threshold: {details.get('Threshold used in PSI')}, Drift: {details.get('Drift status given by PSI')}<br>JS: {details.get('JS Distance statistic')}, Threshold: {details.get('Threshold used in JS Distance')}, Drift: {details.get('Drift status given by JS Distance')}"""
                else:
                    bools = [details.get('KS-Test statistic'), details.get('PSI statistic'), details.get('JS Distance statistic')]
                    color = "rgb(255, 0, 0)" if sum(bools) == 3 else "rgb(255, 165, 0)"
                    hover_text = f"""<b>{feature} Drift Tests</b><br>KS: {details.get('KS-Test statistic')}, p = {details.get('KS-Test p-value')}, Drift: {details.get('Drift status given by KS-Test')}<br>PSI: {details.get('PSI statistic')}, Threshold: {details.get('Threshold used in PSI')}, Drift: {details.get('Drift status given by PSI')}<br>JS: {details.get('JS Distance statistic')}, Threshold: {details.get('Threshold used in JS Distance')}, Drift: {details.get('Drift status given by JS Distance')}"""
                categories['Input Feature Analysis']['Data Drift'].append((feature, hover_text, color))

        seen_pairs = set()
        rel_drift = input_details.get('Relationship Drift Detected', [])
        for pair in rel_drift:
            sorted_pair = tuple(sorted(pair))
            if sorted_pair not in seen_pairs:
                seen_pairs.add(sorted_pair)
                hover_text = f"Pairwise Drift: {sorted_pair[0]} ↔️ {sorted_pair[1]}"
                categories['Input Feature Analysis']['Copula'].append((f"{sorted_pair[0]}-{sorted_pair[1]}", hover_text))

        data_drift_present = len(categories['Input Feature Analysis']['Data Drift']) > 0
        copula_drift_present = len(categories['Input Feature Analysis']['Copula']) > 0
        if not data_drift_present and not copula_drift_present:
            color = "green"
            summary_label = "no drift"
        elif data_drift_present and copula_drift_present:
            summary_label = "data and copula drift present"
            color = "rgb(255, 0, 0)"
        elif data_drift_present:
            summary_label = "data drift present"
            color = "rgb(255, 165, 0)"
        else:
            summary_label = "copula drift present"
            color = "rgb(255, 165, 0)"
        categories['Input Feature Analysis']['Summary'] = [("Input Drift Summary", summary_label,color)]

        if model_type_lower == "cv":
            summary = results_data['Production Data Summary']
            num_anomalies = summary.get('Number of Anomalies', 0)
            if num_anomalies > 0:
                categories['Data Quality'].append(("Anomalies", f"{num_anomalies} anomalies detected", f"rgb(255,{round(255-(((num_anomalies-1)*100/(100-1))*2.55),2)},0)"))
            unique_res_sizes = summary.get('Number of Unique Resolutions and Sizes found', 0)
            if unique_res_sizes > 0:
                categories['Data Quality'].append(("Image Variance", f"{unique_res_sizes} unique resolutions and sizes found", f"rgb(255,{round(255-(((unique_res_sizes-1)*100/(10-1))*2.55),2)},0)"))

    output_drift = results_data.get('Output Feature Details', {})
    if output_drift.get('Final drift status', False):
        if model_type_lower == 'regression':
            hover_text = f"""<b>Prediction Drift (Numerical)</b><br>
KS: {output_drift.get('KS-Test statistic')}, p = {output_drift.get('KS-Test p-value')}, Drift: {output_drift.get('Drift status given by KS-Test')}<br>
PSI: {output_drift.get('PSI statistic')}, Threshold: {output_drift.get('Threshold used in PSI')}, Drift: {output_drift.get('Drift status given by PSI')}<br>
JS: {output_drift.get('JS Distance statistic')}, Threshold: {output_drift.get('Threshold used in JS Distance')}, Drift: {output_drift.get('Drift status given by JS Distance')}"""
        else:
            hover_text = f"""<b>Prediction Drift (Categorical)</b><br>
Chi-Square: {output_drift.get('Chi Square statistic')}, p = {output_drift.get('Chi Square p-value')}, Drift: {output_drift.get('Drift status given by Chi Square')}<br>
PSI: {output_drift.get('PSI statistic')}, Threshold: {output_drift.get('Threshold used in PSI')}, Drift: {output_drift.get('Drift status given by PSI')}<br>
JS: {output_drift.get('JS Distance statistic')}, Threshold: {output_drift.get('Threshold used in JS Distance')}, Drift: {output_drift.get('Drift status given by JS Distance')}"""
        categories['Prediction Drift'].append(("Target", hover_text))

    metrics = results_data['Production Data Summary']['Metrics']
    metric_abbr = {
        "Accuracy": "Accuracy",
        "F1": "F1",
        "F1 Score": "F1",
        "Precision": "Precision",
        "Recall": "Recall",
        "ROC_AUC": "ROC-AUC",
        "Log Loss": "Log",
        "False Positive Rate": "FPR",
        "True Positive Rate": "TPR",
        "Mean Absolute Error": "MAE",
        "Mean Absolute Percentage Error": "MAPE",
        "Mean Squared Error": "MSE",
        "Root Mean Squared Error": "RMSE",
        "R2 Score": "R²",
        "BLEU": "BLEU",
        "Similarity": "Similarity"
    }
    for metric, value in metrics.items():
        if benchmark_metrics is None or metric not in benchmark_metrics:
            continue
        benchmark = benchmark_metrics[metric]
        if metric == "False Positive Rate":
            delta = (value - benchmark)
        else:
            delta = (benchmark - value)*100/benchmark if metric in ["Mean Absolute Error", "Mean Absolute Percentage Error", "Mean Squared Error","Root Mean Squared Error"] else benchmark - value
        show_change = (
            (metric in ["False Positive Rate", "Mean Absolute Error", "Mean Absolute Percentage Error", "Mean Squared Error","Root Mean Squared Error"] and value > benchmark) or
            (metric not in ["False Positive Rate", "Mean Absolute Error", "Mean Absolute Percentage Error", "Mean Squared Error","Root Mean Squared Error"] and value < benchmark)
        )
        if show_change:
            metric_label = metric_abbr.get(metric, metric)
            if metric in ["False Positive Rate", "Mean Absolute Error", "Mean Absolute Percentage Error", "Mean Squared Error","Root Mean Squared Error"] and value > benchmark:
                hover_text = f"{metric} increased by {delta:.2f}%"
            else:
                hover_text = f"{metric} dropped by {delta:.2f}%"
            categories['Performance Drift'].append((metric_label, hover_text, f"rgb(255,{round(255-delta*2.55,2)},0)"))
    return categories

def get_fishbone(model_type, results, model_name, production_run_date, benchmark_metrics=None):
    
    categories = fishbone_plot(model_type=model_type, model_name=model_name, results_data=results, production_run_date=production_run_date, benchmark_metrics=benchmark_metrics)
    # print(categories)
    model_type_lower = model_type.lower()

    fig = go.Figure()
    num_causes = len([k for k in categories.keys()])
    base_length = 8
    needs_subbone = model_type_lower in ["classification", "regression"] and "Input Feature Analysis" in categories
    spine_padding = 0.15 if needs_subbone else 0

    main_arrow_end_x = base_length / 2 + spine_padding
    main_arrow_start_x = -base_length / 2 - spine_padding
    main_arrow_y = 0

    fig.add_annotation(
        x=main_arrow_end_x, y=main_arrow_y,
        ax=main_arrow_start_x - 11.5, ay=main_arrow_y,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=4,
       arrowcolor="grey"
    )

    fig.add_annotation(
        x=main_arrow_end_x + 1.2, y=main_arrow_y, 
        xref="x", yref="y",
        text="Model Drift",
        showarrow=False,
        font=dict(color="white", size=18),
        bgcolor="purple",
        bordercolor="black",
        borderwidth=1,
        borderpad=10,
        align="center"
    )

    causes = list(categories.keys())
    spacing = (main_arrow_end_x - main_arrow_start_x) / (len(causes) + 0.5)
    cause_length = 10

    for i, cause in enumerate(causes):
        x_base = main_arrow_start_x + spacing * (i + 1)
        y_base = main_arrow_y
        is_top = (i % 2 == 0)
        dy = cause_length * 2 if is_top else -cause_length * 2
        dx = -cause_length * 0.7071
        x_tip = x_base + dx
        y_tip = y_base + dy

        fig.add_annotation(
            x=x_base, y=y_base,
            ax=x_tip , ay=y_tip,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
           arrowcolor="grey"
        )

        fig.add_annotation(
            x=x_tip - 0.1, y=y_tip,
            xref="x", yref="y",
            text=cause,
            showarrow=False,
            font=dict(color="black" , size = 14.25),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            align="center"
        )

        subcauses = categories[cause]
        
#         print(subcauses)

        if isinstance(subcauses, dict):  # Nested structure
            sub_spine_start_x = x_tip - 0.9
            sub_spine_y = y_tip
            sub_spine_end_x = sub_spine_start_x - 6

            fig.add_annotation(
                ax=sub_spine_end_x, ay=sub_spine_y,
                x=sub_spine_start_x if cause == "Data Drift" else sub_spine_start_x - 1, 
                y=sub_spine_y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=3,
               arrowcolor="grey"
            )
            summary = subcauses.get("Summary", [])
            # print(subcauses.items())
            other_subs = [(k, v) for k, v in subcauses.items() if k != "Summary"]
            # print(other_subs)

            for j, (subtype, drift_causes) in enumerate(other_subs):
                y_offset = 9 if subtype in ['Syntax Drift', 'Data Drift'] else -9
                sub_x = sub_spine_end_x
                sub_y = sub_spine_y + y_offset
        
                fig.add_annotation(
                    x=sub_spine_start_x - 2 if subtype in ['Syntax Drift', 'Data Drift'] else sub_spine_start_x - 1.8,
                    y=sub_spine_y,
                    ax=sub_x, ay=sub_y,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.5,
                    arrowwidth=2,
                   arrowcolor="grey"
                )

                fig.add_annotation(
                    x=sub_x - 0.2, y=sub_y,
                    xref="x", yref="y",
                    text=subtype,
                    showarrow=False,
                    font=dict(color="black", size=14),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    align="center"
                )
        
                if not drift_causes:
                    t = 0.5
                    base_x = sub_x + (sub_spine_start_x - 2 - sub_x) * t
                    base_y = sub_y + (sub_spine_y - sub_y) * t
                    sub_line_length = 3
                    fig.add_shape(
                        type="line",
                        x0=base_x, y0=base_y,
                        x1=base_x - sub_line_length, y1=base_y,
                        line=dict(color="green", width=1.5, dash="solid")
                    )
                    fig.add_trace(go.Scatter(
                        x=[base_x - sub_line_length / 2],
                        y=[base_y + 0.35],
                        mode="text",
                        text=["No Drift"],
                        textfont=dict(color="black", size=13),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    continue
                max_visible = 3
                overflow_causes = drift_causes[max_visible:] if len(drift_causes) > max_visible else []
                causes_to_show = drift_causes[:max_visible]
        
                total_display = causes_to_show + ([("More Issues", "<br>".join([s for s, _ in overflow_causes]))] if overflow_causes else [])
        
                for k, text in enumerate(total_display):
                    print(text)
                    short_text, hover_text,line_color = text[0], text[1], text[2] if len(text) == 3 else "grey"
                    t = (k + 1) / (len(total_display) + 1)
                    base_x = sub_x + (sub_spine_start_x - 2 - sub_x) * t
                    base_y = sub_y + (sub_spine_y - sub_y) * t
                    sub_line_length = min(max(len(short_text) * 0.15, 3), 8)
                    fig.add_shape(
                        type="line",
                        x0=base_x, y0=base_y,
                        x1=base_x - sub_line_length, y1=base_y,
                        line=dict(color=line_color, width=1.5)
                    )
                    fig.add_trace(go.Scatter(
                        x=[base_x - sub_line_length / 2],
                        y=[base_y + 0.35],
                        mode="text",
                        text=[short_text],
                        hovertext=[hover_text],
                        hoverinfo="text",
                        textfont=dict(color="black", size=11.75),
                        showlegend=False
                    ))

            if summary:
                short_text, hover_text,color = summary[0]  # Only one summary item
                x_on_arrow = x_tip + (x_base - x_tip) * 0.5
                y_on_arrow = y_tip + (y_base - y_tip) * 0.5
                sub_line_length = min(max(len(hover_text) * 0.15, 3), 8)

                fig.add_shape(
                    type="line",
                    x0=x_on_arrow, y0=y_on_arrow,
                    x1=x_on_arrow - sub_line_length, y1=y_on_arrow,
                    line=dict(color=color, width=1.5)
                )
                fig.add_trace(go.Scatter(
                    x=[x_on_arrow - sub_line_length / 2],
                    y=[y_on_arrow + 0.35],
                    mode="text",
                    text=[hover_text],
                    hovertext=[hover_text],
                    hoverinfo="text",
                    textfont=dict(color="black", size=13),
                    showlegend=False
                ))


        elif model_type_lower == "natural language processing (nlp)" and cause == "Data Quality":
            
            combined_hover = categories["Data Quality"][0][1]
            color = categories["Data Quality"][0][2]

            x_on_arrow = x_tip + (x_base - x_tip) * 0.5
            y_on_arrow = y_tip + (y_base - y_tip) * 0.5
            sub_line_length = 3

            fig.add_shape(
                type="line",
                x0=x_on_arrow, y0=y_on_arrow,
                x1=x_on_arrow - sub_line_length, y1=y_on_arrow,
                line=dict(color=color, width=1.5)
            )
            fig.add_trace(go.Scatter(
                x=[x_on_arrow - sub_line_length / 2],
                y=[y_on_arrow + 0.35],
                mode="text",
                text=["text"],
                hovertext=[combined_hover],
                hoverinfo="text",
                textfont=dict(color="black", size=13.25),
                showlegend=False
            ))
        else:
            if len(subcauses) == 0:
                x_on_arrow = x_tip + (x_base - x_tip) * 0.5
                y_on_arrow = y_tip + (y_base - y_tip) * 0.5
                sub_line_length = 3
                fig.add_shape(
                    type="line",
                    x0=x_on_arrow, y0=y_on_arrow,
                    x1=x_on_arrow - sub_line_length, y1=y_on_arrow,
                    line=dict(color="green", width=1.5, dash="solid")
                )
                fig.add_trace(go.Scatter(
                    x=[x_on_arrow - sub_line_length / 2],
                    y=[y_on_arrow + 0.35],
                    mode="text",
                    text=["No Drift"],
                    textfont=dict(color="black", size=13.25),
                    showlegend=False,
                    hoverinfo='text'
                ))
            else:
                for j, subcause in enumerate(subcauses):
                    # print(subcause)
                    frac = (j + 1) / (len(subcauses) + 0.5)
                    x_on_arrow = x_tip + (x_base - x_tip) * frac
                    y_on_arrow = y_tip + (y_base - y_tip) * frac
                    if isinstance(subcause, tuple):
                        short_text, hover_text,color = subcause
                    else:
                        short_text = subcause.split('\n')[0]
                        hover_text = subcause
                    sub_line_length = min(max(len(short_text) * 0.15, 3), 8)
                    fig.add_shape(
                        type="line",
                        x0=x_on_arrow, y0=y_on_arrow,
                        x1=x_on_arrow - sub_line_length, y1=y_on_arrow,
                        line=dict(color=color, width=1.5)
                    )
                    fig.add_trace(go.Scatter(
                        x=[x_on_arrow - sub_line_length / 2],
                        y=[y_on_arrow + 0.35],
                        mode="text",
                        text=[short_text],
                        hovertext=[hover_text],
                        hoverinfo="text",
                        textfont=dict(color="black", size=13.25),
                        showlegend=False
                    ))
                    

    fig.update_layout(
        xaxis=dict(range=[-20, 8], showgrid=False, visible=False),
        yaxis=dict(range=[-30, 23], showgrid=False, visible=False),
        width=1500,
        height=900
    )
    return fig

def create_drift_indicator(model_name, model_type, production_run_date):
    
    fig = go.Figure()

    # Simulate vertical gradient using colored rectangles (bottom to top)
    colors = ['#FFFF00', '#FFCC00', '#FF9900', '#FF6600', '#FF3300', '#FF0000']
    n_segments = len(colors)

    for i, color in enumerate(colors):
        # print(i)
        fig.add_shape(
            type="rect",
            x0 = (i+1) / n_segments,
            x1= (i+2) / n_segments,
            y0=0.6,
            y1=0.8,
            fillcolor=color,
            line=dict(width=0),
        )

    fig.add_trace(go.Scatter(
        x=[1.3],
        y=[0.7],
        mode='markers',
        marker=dict(size=16, color='green', symbol='square'),
        name='No Drift'
    ))

    # Add drift zone labels
    fig.add_annotation(x=0.15, y=0.4, text="Low Drift", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=0.4, y=0.4, text="Low to Moderate Drift", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=0.65, y=0.4, text="Moderate Drift", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=0.9, y=0.4, text="Moderate to High Drift", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=1.15, y=0.4, text="High Drift", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=1.3, y=0.4, text="No Drift", showarrow=False, font=dict(size=12))

    # Update layout
    fig.update_layout(
        title= f"{model_name} - {model_type}<br><sup>Production Run: {production_run_date}</sup>" if model_type.lower() != 'text' else f"{model_name} - Natural Language Processing (NLP) Model Drift<br><sup>Production Run: {production_run_date}</sup>",
        xaxis=dict(range=[0, 1.5], visible=False),
        yaxis=dict(range=[0, 1.5], visible=False),
        height=150,
        width=800,
        margin=dict(l=0, r=0, t=40, b=10),
        showlegend=False
    )
    return fig
 

    
def get_ollama_template(model_type):
    
    # llm_name = "_".join(model_name.lower().split())
    
    if model_type in ["Classification", "Regression"]:
        
        system = """You are a very smart, knowledgeable, and helpful assistant that answers questions related to model degradation issues, metrics, and data of a {model_name} application, which is a {model_type} task."

    You are part of a root cause analysis application where machine learning models are deployed and assessed against 5 analyses types:
    1) Performance Drift Analysis: Compare model performance metrics (e.g., accuracy, precision, recall) on current production run with ground truths
    2) Prediction Drift Analysis: Perform tests like KS-test, Chi-square test, PSI and JS on output label to analyze drift
    3) Data Drift Analysis: Perform tests like KS-test, Chi-square test, PSI and JS on input data to analyze drift
    4) Data Quality Analysis: Analyze data quality metrics such as missing values, duplicates, business rules, datatype mismatches and outliers
    5) Model Explanations/Interpretations: Use SHAP to provide explanations for each instance in the data using the model

    You'll be provided domain knowledge about the model, its baseline data, and production data (based on the production date).

    Domain Knowledge:
    {domain_knowledge}

    Baseline Information:
    {baseline_stats}
        
    Instructions:
    1) DO NOT provide false information. Answer only based on available data.
    2) Provide ONLY accurate information about metrics or clinical jargon.
    3) Keep answers concise (≤30 words).
    4) Always be polite, ethical, and safe. NO harmful, illegal, or offensive responses.
    5) If you don’t know the answer, admit it instead of guessing.
    6) Suggest potential root causes for model degradation while following all other guidelines.
    7) Whenever asked a reason for a problem (e.g root cause) in any analysis, make sure to relate a VALID problem from the production run's results and the domain knowledge, baseline information

    Your name is {llm_name}_mistralLLM.
    You are a large language model that is very helpful and knowledgeable about the {model_name} application.
    You'll be given information about the analysis results for the production run in the prompt.
    
    {human_prompt}
    
    """
    
    if model_type == "Natural Language Processing (NLP)":
        
        system = """You are a very smart, knowledgeable, and helpful assistant that answers questions related to model degradation issues, metrics, and data of a {model_name} application, which is a {model_type} task, giving outputs as {output_type}"

    You are part of a root cause analysis application where machine learning models are deployed and assessed against 5 analyses types:
    1) Performance Drift Analysis: Compare model performance metrics (e.g., accuracy, precision, recall) on current production run with ground truths
    2) Prediction Drift Analysis: Perform tests like Chi-square test, PSI and JS on output label to analyze drift
    3) Data Drift Analysis: Perform tests like KS-test, Chi-square test, PSI and JS on features extracted from input images to analyze drift
    4) Data Quality Analysis: Analyze data quality metrics such as sharpness, brightness, noise, resolution, size, and anomalies
    5) Model Explanations/Interpretations: Use LIME to provide explanations for each instance in the data using the model

    You'll be provided domain knowledge about the model, its baseline data, and production data (based on the production date).

    Domain Knowledge:
    {domain_knowledge}

    Baseline Information:
    {baseline_stats}
        
    Instructions:
    1) DO NOT provide false information. Answer only based on available data.
    2) Provide ONLY accurate information about metrics or clinical jargon.
    3) Keep answers concise (≤30 words).
    4) Always be polite, ethical, and safe. NO harmful, illegal, or offensive responses.
    5) If you don’t know the answer, admit it instead of guessing.
    6) Suggest potential root causes for model degradation while following all other guidelines.
    7) Whenever asked a reason for a problem (e.g root cause) in any analysis, make sure to relate a VALID problem from the production run's results and the domain knowledge, baseline information


    Your name is {llm_name}_mistralLLM.
    You are a large language model that is very helpful and knowledgeable about the {model_name} application.
    You'll be given information about the analysis results for the production run in the prompt.
    
    {human_prompt}
    
    """
    
    
    
    if model_type == "Computer Vision (CV)":
        
        system = """You are a very smart, knowledgeable, and helpful assistant that answers questions related to model degradation issues, metrics, and data of a {model_name} application, which is a {model_type}, Image Classification task."

    You are a Computer Vision model, the input for the model is an image and features are extracted from the image.
    
    You are part of a root cause analysis application where machine learning models are deployed and assessed against 5 analyses types:
    1) Performance Drift Analysis: Compare model performance metrics (e.g., accuracy, precision, recall) on current production run with ground truths
    2) Prediction Drift Analysis: Perform tests like Chi-square test, PSI and JS on output label to analyze drift
    3) Data Drift Analysis: Perform tests like KS-test, Chi-square test, PSI and JS on features extracted from input images to analyze drift
    4) Data Quality Analysis: Analyze data quality metrics such as sharpness, brightness, noise, resolution, size, and anomalies
    5) Model Explanations/Interpretations: Use LIME to provide explanations for each instance in the data using the model

    You'll be provided domain knowledge about the model, its baseline data, and production data (based on the production date).

    Domain Knowledge:
    {domain_knowledge}

    Baseline Information:
    {baseline_stats}
        
    Instructions:
    1) DO NOT provide false information. Answer only based on available data.
    2) Provide ONLY accurate information about metrics or clinical jargon.
    3) Keep answers concise (≤30 words).
    4) Always be polite, ethical, and safe. NO harmful, illegal, or offensive responses.
    5) If you don’t know the answer, admit it instead of guessing.
    6) Suggest potential root causes for model degradation while following all other guidelines.
    7) Whenever asked a reason for a problem (e.g root cause) in any analysis, make sure to relate a VALID problem from the production run's results and the domain knowledge, baseline information

    Your name is {llm_name}_mistralLLM.
    You are a large language model that is very helpful and knowledgeable about the {model_name} application.
    You'll be given information about the analysis results for the production run in the prompt.
    
    {human_prompt}
    
    """

    
    return system
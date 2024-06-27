
# Description:

This module is a python implementation of the optimal subgaussian mean estimator
from

J. C. H. Lee and P. Valiant, "Optimal Sub-Gaussian Mean Estimation in  R"
2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS),
Denver, CO, USA, 2022, pp. 672-683, doi: 10.1109/FOCS52979.2021.00071.
https://arxiv.org/abs/2011.08384

As the provided examples show, the estimator is superior to the ordinary sample mean X=1/x.size*sum_i x_i for an array with
elements x_i and size x.size:

for symmetric heavily tailed distributions, like the T or the Laplace distributions.
In 1000 trials, the optimal estimator is better or equal for roughly 647 trial runs
and the sample mean is better in just 350 trials.

For standard normal distributions, a few executions of the last example show that the estimator
is somewhat equal to the population mean (within the supplied confidence interval delta and
apart from numerical floating point imprecisions). In 1000 trial runs, the optimal estimator mean is
better or equal than the sample mean in 513 runs and in 484 cases sample mean is equal or better.


Unfortunately, it appears that for skewed distributions, problems can arise.

For the inverse gamma distribution and 1000 mean estimations, the sample mean is better or equal than the sample mean
in 504 cases, while the optimal mean estimator is better or equal in 493 cases.

It appears that this has something to do with the asymmetry. The exponential distribution is even more
skewed than the inverse Gamma distribution. In 1000 trial runs, the sample mean is better or equal in
517 cases and than the optimal estimator is better in 480 cases for the exponential distribution.

More trials seem to establish the picture that there is a perhaps numerical problem with skewed distributions.


The implementation consists of a function

mean

it computes the optimal mean estimator for numpy arrays.

it expects a numpy array a, a confidence parameter delta and its other
arguments match the behavior of the numpy.mean function whose documentation is
given here:
https://numpy.org/doc/stable/reference/generated/numpy.mean.html



The computed mean estimator is by default a numpy f64 value if the array is of
integer type. Otherwise, the result is of the same type as the array, which is
also similar as the numpy mean function.

The estimator is computed to fulfill the equation

P(|mu-mean|<epsilon)>=1-delta

by default, delta=0.05.



The module also has a function

mean_flattened

This function works in the same way as optimal_mean_estimator, but it flattens the arrays that it recieves
and also has no optional out parameter. Instead of a matrix, it returns single int, float or complex values.

# Definitions:

## for arrays in arbitrary shapes:

def mean (a,delta=0.01, axis=None,dtype=None,out=None,keepdims=False,where=None):
    """
    Computes the optimal mean estimator of

    J. C. H. Lee and P. Valiant, "Optimal Sub-Gaussian Mean Estimation in  R"
    2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS),
    Denver, CO, USA, 2022, pp. 672-683, doi: 10.1109/FOCS52979.2021.00071.
    https://arxiv.org/abs/2011.08384


    for numpy arrays.

    #Args:
        a: A numpy array whose mean is computed. can by of integer, floating or
        complex type and have arbitrary axes.

        delta: a float value. The computed estimator fulfills
                P(|mu-mean|<epsilon)>=1-delta. The default is 0.001. Note that
                the size of the sample must be larger than ln(1/delta).

        axis:  a tuple defining the axes over which the mean is to be computed,
               default isnone

        dtype:  numpy type of the computed result. Per default, if a is an
                integer array, a numpy.f64 is returned, otherwise, unless
                specified explicitly, the default return is of the same type
                as the array a.
                Note that if an int type is specified, float64 values are used
                during the computation in order to increase precision.

        out:    a numpy array that stores the means.
                If the result is a matrix out must have the same shape
                as the computed result. If the result is a single mean value, the
                output is stored in out[0].

        keepdims: boolean, the default is false. If  set to True, the reduced axes
                are left in the result as dimensions with size one and the
                result broadcasts against the input array.
        where: a boolean numpy array. Elements set to false are discarded from the
                computation

    Returns:
        The computed mean estimator is by default a numpy f64 value if the array is of
        integer type. Otherwise, the result is of the same type as the array, which is
        also similar as the numpy mean function.

        The estimator is computed to fulfill the equation

        P(|mu-mean|<epsilon)>=1-delta
    """

    """ if no type is supplied and the array is integer, output is np.float64,
        otherwise it is the array's datatype if no type was supplied. if a type
        was supplied the output is that specified datatype"""


and:

## For flattened arrays

def mean_flattened(x, delta=0.05,dtype=None,where=None):
    """
    Computes the optimal mean estimator of

    J. C. H. Lee and P. Valiant, "Optimal Sub-Gaussian Mean Estimation in  R"
    2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS),
    Denver, CO, USA, 2022, pp. 672-683, doi: 10.1109/FOCS52979.2021.00071.
    https://arxiv.org/abs/2011.08384


    for flattened numpy arrays with one axis. It is slightly faster than
    optimal_mean_estimator because it does not make computations to determine
    the axes of the result.

    Args:
        a: A numpy array whose mean is computed. can by of integer, floating or
        complex type and have arbitrary axes.

        delta: a float value. The computed estimator fulfills
               P(|mu-mean|<epsilon)>=1-delta. The default is 0.001. Note that
               the size of the sample must be larger than ln(1/delta).


        dtype:  numpy type of the computed result. Per default, if a is an
                integer array, a numpy.f64 is returned, otherwise, unless
                specified explicitly, the default return is of the same type
                as the array a.
                Note that if an int type is specified, float64 values are used
                during the computation as intermediate values,
                in order to increase precision of the casted final results.

        where: a boolean numpy array. Elements set to false are discarded from the
                computation

# Examples:
## Example 1

    import numpy as np

    import optimal_mean_estimator.optimalmean as ope

    x = np.arange(1*2*3*4)

    x=np.reshape(x,(1,2,3,4))

    axes=(2,3,0)


    ordinary_mean=np.mean(x,axes)

    print(ordinary_mean)

    optimal_mean=ope.mean(x,0.001,axis=axes)

    print(optimal_mean)

## Example 2

    import numpy as np

    import optimal_mean_estimator.optimalmean as ope

    x = np.arange(1*2*3*4)

    x=np.reshape(x,(1,2,3,4))

    axes=(2,3,0)


    mask = np.zeros(1*2*3*4)

    mask[:3] = 1

    np.random.shuffle(mask)

    mask = np.reshape(x,(1,2,3,4))

    mask = mask.astype(bool)

    ordinary_mean=np.mean(x,axes,where=mask,keepdims=True)

    print(ordinary_mean)

    optimal_mean=ope.mean(x,0.001,axis=axes,keepdims=True,where=mask)

    print(optimal_mean)

## Example 3

    import numpy as np

    import optimal_mean_estimator.optimalmean as ope

    x = np.arange(1*2*3*4)

    x=np.reshape(x,(1,2,3,4))

    axes=(2,1)

    ordinary_mean=np.mean(x,axes,dtype=np.int32)

    print(ordinary_mean)

    optimal_mean=ope.mean(x,0.001,axis=axes,dtype=np.int32)

    print(optimal_mean)


## Example 4

This example shows that the estimator works for heavy tailed distribution like the T or the Laplace distributions.
If executed several times, it also shows that, appart from small numerical floating point problems, for symmetric distributions
that are not heavily tailed, the estimator yields similarly good results as the ordinary sample mean.

However, for heavily skewed distributions, like the inverse Gamma, or worse the exponential distributions, the ordinary sample mean seems to
perform sometimes  better by a small amount.

    import numpy as np

    import optimal_mean_estimator.optimalmean as ope

    import matplotlib.pyplot as plt

    i=0

    p=0

    c=0

    error1=np.zeros(shape=(1000))

    error2=np.zeros(shape=(1000))

    result1=np.zeros(shape=(1000))

    result2=np.zeros(shape=(1000))

    populationmean=np.zeros(shape=(1000))
    
    skewness=np.zeros(shape=(1000))

    userinput= int(input("""put in 1 for a T distribution with df=3

                     \n 2 for a normal distribution

                     \n 3 for a Laplace distribution

                     \n 4 for an inverse Gamma Verteilung

                     \n 5 for an exponential distribution

                     \n default is 1 \n\n\n"""))


    import random

    while(i<1000):

        match userinput:

            case 1: population = np.random.standard_t(df=3, size=100000)

            case 2: population= np.random.standard_normal(size=100000)

            case 3: population=np.random.laplace(size=100000)

            case 4: population=np.random.gamma(4.0, scale=1.0, size=100000)

            case 5: population=np.random.exponential(scale=1.0, size=100000)

            case _: population = np.random.standard_t(df=3, size=100000)

        populationmean[i]=np.mean(population)
        skewness[i]=scipy.stats.skew(population)
        
        sample = np.array(random.choices(population, k=1000))
        
        
        result1[i]=np.mean(sample)

        error1[i]=(populationmean[i]-result1[i])**2

        result2[i]=ope.mean_flattened(sample,0.001)

        error2[i]=(populationmean[i]-result2[i])**2

        if error1[i]<=error2[i]:p=p+1

        if error2[i]<=error1[i]:c=c+1

        i=i+1


    match userinput:

        case 1: print("\n\n for a T distribution with df=3 \n")

        case 2: print(" 2 for a normal distribution \n")

        case 3: print(" 3 for a Laplace distribution \n")

        case 4: print(" 4 for an inverse Gamma distribution \n")

        case 5: print(" 5 for an exponential distribution \n")

        case _: print(" for a T distribution with df=3 \n ")


    print(p," times the sample mean was better or equal than the optimal mean estimator \n",
        "\n average squared error to the populationmean ",
        np.mean(error1), "\n\n", )

    print(c, " times the optimal mean estimator was better or equal than the sample mean\n" ,
        "\n average squared error to the populationmean ",
        np.mean(error2),"\n\n")
        
    print("mean skewness of the population", np.mean(skewness))
    
    plt.hist(population, bins='auto')  # arguments are passed to np.histogram
    plt.title("distribution Histogram")
    plt.show()

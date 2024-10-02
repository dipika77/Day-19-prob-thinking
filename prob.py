#Thinking probabilistically: discrete variable
#Random number generators and hacker statistics

#instructions
'''Instantiate and seed a random number generator, rng, using the seed 42.
Initialize an empty array, random_numbers, of 100,000 entries to store the random numbers. Make sure you use np.empty(100000) to do this.
Write a for loop to draw 100,000 random numbers using rng.random(), storing them in the random_numbers array. To do so, loop over range(100000).
Plot a histogram of random_numbers. It is not necessary to label the axes in this case because we are just checking the random number generator. Hit submit to show your plot.'''

# Instantiate and seed the random number generator
reng = np.random.default_rng(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = reng.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()



#instructions
'''Define a function with signature perform_bernoulli_trials(n, p).
Initialize to zero a variable n_success the counter of Trues, which are Bernoulli trial successes.
Write a for loop where you perform a Bernoulli trial in each iteration and increment the number of success if the result is True. Perform n iterations by looping over range(n).
To perform a Bernoulli trial, choose a random number between zero and one using rng.random(). If the number you chose is less than p, increment n_success (use the += 1 operator to achieve this). An RNG has already been instantiated as the variable rng and seeded.
The function returns the number of successes n_success'''

import numpy as np
rng = np.random.default_rng(42)
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = rng.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success



'''Seed the random number generator to 42.
Initialize n_defaults, an empty array, using np.empty(). It should contain 1000 entries, since we are doing 1000 simulations.
Write a for loop with 1000 iterations to compute the number of defaults per 100 loans using the perform_bernoulli_trials() function. It accepts two arguments: the number of trials n - in this case 100 - and the probability of success p - in this case the probability of a default, which is 0.05. On each iteration of the loop store the result in an entry of n_defaults.
Plot a histogram of n_defaults. Include the density=True keyword argument so that the height of the bars of the histogram indicate the probability.
Show your plot.'''

# Instantiate and seed random number generator
rng = np.random.default_rng(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100,0.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults,density = True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()


#instructions
'''Compute the x and y values for the ECDF of n_defaults.
Plot the ECDF, making sure to label the axes. Remember to include marker = '.' and linestyle = 'none' in addition to x and y in your call plt.plot().
Show the plot.
Compute the total number of entries in your n_defaults array that were greater than or equal to 10. To do so, compute a boolean array that tells you whether a given entry of n_defaults is >= 10. Then sum all the entries in this array using np.sum(). For example, np.sum(n_defaults <= 5) would compute the number of defaults with 5 or fewer defaults.
The probability that the bank loses money is the fraction of n_defaults that are greater than or equal to 10. Print this result by hitting submit!'''

# Compute ECDF: x, y
x,y = ecdf(n_defaults)

# Plot the ECDF with labeled axes
plt.plot(x,y, marker = '.', linestyle = 'none')
plt.xlabel('number of defaults out of 100')
plt.ylabel('ECDF')

# Show the plot
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
n_lose_money = np.sum(n_defaults >= 10)

# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))



#instructions
'''Draw samples out of the Binomial distribution using rng.binomial(). You should use parameters n = 100 and p = 0.05, and set the size keyword argument to 10000.
Compute the CDF using your previously-written ecdf() function.
Plot the CDF with axis labels. The x-axis here is the number of defaults out of 100 loans, while the y-axis is the CDF.
Show the plot.'''

# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = rng.binomial(100,0.05, size = 10000)

# Compute CDF: x, y
x,y = ecdf(n_defaults)

# Plot the CDF with axis labels
plt.plot(x,y, marker = '.', linestyle = 'none')
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('CDF')

# Show the plot
plt.show()



#instructions
'''Using np.arange(), compute the bin edges such that the bins are centered on the integers. Store the resulting array in the variable bins.
Use plt.hist() to plot the histogram of n_defaults with the density=True and bins=bins keyword arguments.
Show the plot.'''

# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
plt.hist(n_defaults, density = True, bins = bins)

# Label axes
plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('ECDF')

# Show the plot
plt.show()



#instructions
'''Using the rng.poisson() function, draw 10000 samples from a Poisson distribution with a mean of 10.
Make a list of the n and p values to consider for the Binomial distribution. Choose n = [20, 100, 1000] and p = [0.5, 0.1, 0.01] so that 
 is always 10.
Using rng.binomial() inside the provided for loop, draw 10000 samples from a Binomial distribution with each n, p pair and print the mean and standard deviation of the samples. There are 3 n, p pairs: 20, 0.5, 100, 0.1, and 1000, 0.01. These can be accessed inside the loop as n[i], p[i].'''


# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = rng.poisson(10, size = 10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20,100,1000]
p = [0.5,0.1,0.01]


# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = rng.binomial(n[i], p[i], size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))



#instructions
'''Draw 10000 samples from a Poisson distribution with a mean of 251/115 and assign to n_nohitters.
Determine how many of your samples had a result greater than or equal to 7 and assign to n_large.
Compute the probability, p_large, of having 7 or more no-hitters by dividing n_large by the total number of samples (10000).
Hit submit to print the probability that you calculated.'''

# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = rng.poisson(251/115, size = 10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large / 10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)



#probability density function
#instructions
'''Draw 100,000 samples from a Normal distribution that has a mean of 20 and a standard deviation of 1. Do the same for Normal distributions with standard deviations of 3 and 10, each still with a mean of 20. Assign the results to samples_std1, samples_std3 and samples_std10, respectively.
Plot a histogram of each of the samples; for each, use 100 bins, also using the keyword arguments density=True and histtype='step'. The latter keyword argument makes the plot look much like the smooth theoretical PDF. You will need to make 3 plt.hist() calls.
Hit submit to make a legend, showing which standard deviations you used, and show your plot! There is no need to label the axes because we have not defined what is being described by the Normal distribution; we are just looking at shapes of PDFs.'''

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = rng.normal(20,1, size = 100000)
samples_std3 = rng.normal(20,3, size = 100000)
samples_std10 = rng.normal(20,10, size = 100000)

# Make histograms
plt.hist(samples_std1, bins = 100, density = True, histtype = 'step')
plt.hist(samples_std3, bins = 100, density = True, histtype = 'step')
plt.hist(samples_std10, bins = 100, density = True, histtype = 'step')

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()


#instructions
'''Use your ecdf() function to generate x and y values for CDFs: x_std1, y_std1, x_std3, y_std3 and x_std10, y_std10, respectively.
Plot all three CDFs as dots (do not forget the marker and linestyle keyword arguments!).
Hit submit to make a legend, showing which standard deviations you used, and to show your plot. There is no need to label the axes because we have not defined what is being described by the Normal distribution; we are just looking at shapes of CDFs.'''

# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

# Plot CDFs
_ = plt.plot(x_std1, y_std1, marker='.', linestyle='none')
_ = plt.plot(x_std3, y_std3, marker='.', linestyle='none')
_ = plt.plot(x_std10, y_std10, marker='.', linestyle='none')



# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()


#instructions
'''Compute mean and standard deviation of Belmont winners' times with the two outliers removed. The NumPy array belmont_no_outliers has these data.
Take 10,000 samples out of a normal distribution with this mean and standard deviation using rng.normal().
Compute the CDF of the theoretical samples and the ECDF of the Belmont winners' data, assigning the results to x_theor, y_theor and x, y, respectively.
Hit submit to plot the CDF of your samples with the ECDF, label your axes and show the plot.'''

# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)


# Sample out of a normal distribution with this mu and sigma: samples
samples = rng.normal(mu,sigma, size = 10000)

# Get the CDF of the samples and of the data
x, y = ecdf(belmont_no_outliers)
x_theor,y_theor = ecdf(samples)


# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()


#instructions
'''Take 1,000,000 samples from the normal distribution using the rng.normal() function. The mean mu and standard deviation sigma are already loaded into the namespace of your IPython instance.
Compute the fraction of samples that have a time less than or equal to Secretariat's time of 144 seconds.
Print the result.'''

# Take a million samples out of the Normal distribution: samples
samples = rng.normal(mu,sigma,1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples <= 144) / len(samples)

# Print the result
print('Probability of besting Secretariat:', prob)


#instructions
'''Define a function with call signature successive_poisson(tau1, tau2, size=1) that samples the waiting time for a no-hitter and a hit of the cycle.
Draw waiting times (size number of samples) for the no-hitter out of an exponential distribution parametrized by tau1 and assign to t1.
Draw waiting times (size number of samples) for hitting the cycle out of an exponential distribution parametrized by tau2 and assign to t2.
The function returns the sum of the waiting times for the two events.'''

def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = rng.exponential(tau1, size = size)

    # Draw samples out of second exponential distribution: t2
    t2 = rng.exponential(tau2, size = size)

    return t1 + t2


#instructions
'''Use your successive_poisson() function to draw 100,000 out of the distribution of waiting times for observing a no-hitter and a hitting of the cycle.
Plot the PDF of the waiting times using the step histogram technique of a previous exercise. Don't forget the necessary keyword arguments. You should use bins=100, density=True, and histtype='step'.
Label the axes.
Show your plot.'''

# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764,715, size = 100000)

# Make the histogram
plt.hist(waiting_times,bins = 100, density = True, histtype = 'step')

# Label axes
plt.xlabel('total waiting time (games)')
plt.ylabel('PDF')

# Show the plot
plt.show()
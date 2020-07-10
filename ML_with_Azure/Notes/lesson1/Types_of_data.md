Big data has become part of the lexicon of organizations worldwide, as more and more organizations look to leverage data to drive informed business decisions. With this evolution in business decision-making, the amount of raw data collected, along with the number and diversity of data sources, is growing at an astounding rate. This data presents enormous potential.

Raw data, however, is often noisy and unreliable and may contain missing values and outliers. Using such data for modeling can produce misleading results. For the data scientist, the ability to combine large, disparate data sets into a format more appropriate for analysis is an increasingly crucial skill.

The data science process typically starts with collecting and preparing the data before moving on to training, evaluating, and deploying a model.

All data in machine learning eventually ends up being numerical, regardless of whether it is numerical in its original form, so it can be processed by machine learning algorithms.

For example, we may want to use gender information in the dataset to predict if an individual has heart disease. Before we can use this information with a machine learning algorithm, we need to transfer male vs. female into numbers, for instance, 1 means a person is male and 2 means a person is female, so it can be processed. Note here that the value 1 or 2 does not carry any meaning.

Another example would be using pictures uploaded by customers to identify if they are satisfied with the service. Pictures are not initially in numerical form but they will need to be transformed into RGB values, a set of numerical values ranging from 0 to 255, to be processed.

## Tabular Data
In machine learning, the most common type of data you'll encounter is tabular data—that is, data that is arranged in a data table. This is essentially the same format as you work with when you look at data in a spreadsheet.

Here's an example of tabular data showing some different clothing products and their properties:

|SKU | Make |	Color |	Quantity | Price |
|--- | ---- | ----- | -------- | ----- |
| 908721 | Guess | Blue | 789	| 45.33 |
| 456552 | Tillys |	Red |	244	|22.91 |
| 789921 | A&F | Green	|387	|25.92 |
| 872266 | Guess | Blue	|154 |	17.56 |

Notice how tabular data is arranged in rows and columns.

## Vectors
It is important to know that in machine learning we ultimately always work with numbers or specifically vectors.

*A vector is simply an array of numbers, such as (1, 2, 3)—or a nested array that contains other arrays of numbers, such as (1, 2, (1, 2, 3)).*

For now, the main points you need to be aware of are that:

All non-numerical data types (such as images, text, and categories) must eventually be represented as numbers
In machine learning, the numerical representation will be in the form of an array of numbers—that is, a vector
As we go through this course, we'll look at some different ways to take non-numerical data and vectorize it (that is, transform it into vector form).

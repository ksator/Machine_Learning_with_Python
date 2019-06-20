import seaborn as sns
import matplotlib.pyplot as plt

# load the iris dataset
iris = sns.load_dataset("iris")

# visualize the relationship between the 4 features for each of three species of Iris
sns.pairplot(iris, hue='species', height=1.5)
plt.show()

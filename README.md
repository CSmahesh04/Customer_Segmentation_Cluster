 <h1 align='center'>CUSTOMER SEGMENTATION USING CLUSTERING TECHNIQUES</h1> 

AI is a hugely important and lucrative endeavour in commerce, helping businesses in optimising products, planning inventory and logistics, along with more technical aspects such as Computer Vision, Natural Language Processing and Recommendation Systems. A lesser known use of AI is its use in Marketing. AI is often used in marketing efforts where speed is essential. AI tools use data and customer profiles to learn how to best communicate with customers, then serve them tailored messages at the right time without intervention from marketing team members, ensuring maximum efficiency. In this project I will apply traditional ML techniques along with DL networks to segment customers of a business so that they can easily advertise to those particular groups of customers. 

The Dataset contains extensive information about customers from a retail analytics company based in Seattle, USA for a period of 2.5 years. The dataset can be found at the following link: https://www.kaggle.com/kyanyoga/sample-sales-data

## Technologies Used

<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
 
 * **Python**
 * **Pandas**
 * **Numpy**
 * **Seaborn**
 * **Matplotlib**
 * **Plotly**
 * **CV2**
 * **Tensorflow 2.0**
 * **Keras**
 * **Sci-kit Learn**
 * **Google Collab**
 </details>
 
 ## About the Data
 
The dataset contains 2,823 rows and 25 columns. The 25 columns contain order information such as order number, price, order date and status of the order etc. It also contains information about the customer such as the address, state, country, phone number etc. Due to such a wide variety of columns the dataset contains many datatypes. I converted the order date from the _object_ datatype to _datetime64_ datatype using a pandas module. One good practice is to actually take a look at the dataset, this is called a sanity check. Below you can see all the datatypes and the total number of null values in each column.
 
 <h5 align="center">Datatypes and Number of Null Values in Dataset</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/Capture.PNG" width=600>
</p>
 
Since there are more than 50% of the values missing in the columns Addressline2 and State, it is best if we drop these columns. Along with these, columns such as Addressline1, Phone, Contact First Name, Contact last Name, Customer Name, Postal Code and Order Number are not useful in segmenting customers based on their features. So we drop these columns as well. Now we have a clean dataset which contains useful features for finding a pattern among customers.
 
 ## EDA and Cleaning
 
 The territory column interested me. The unique values of that column are EMEA, Japan and APAC, along with null values. Upon further inspection I found out that all null value rows in column Territory are orders from countries USA or Canada. So I replaced the null values with CUSA, the Canada-US trade region. With this there are no null values in our dataset. I then use the **Plotly** library to make interactive bar graphs for a few columns to see the distribution among them. We can see that the **Status** column is heavily disproportionate with the 'Shipped" status having 20X more entries than all the rest combined. Such imbalanced features can ruin the performance of a model, so the **Status** column is dropped. Below are the bar graphs:
 
 **PICS OF ALL THE BAR GRPAHS**
 <h5 align="center">Categorical Features</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/bar1.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/bar2.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/bar3.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/bar4.png" width=600>
</p>

 All the above columns are categorical variables. We need to replace the string values with one hot encoded values. Do achieve this I utilized the _get_dummies_ module in **Pandas**. This increased the total number of columns from 9 to 39. The **PRODUCTCODE** column has 109 unique values, so using the same method will give an additional 108 columns which is bad, we need to avoid the curse of dimensionality. Below is the dataset after converting columns into dummies:
 
 <h5 align="center">Dataset with Dummified Features</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/dataset_dummy.PNG" width=600>
</p>
 
 Next I wanted to check how the sales were divided based on the column **ORDERDATE**. From the below line graph it can be seen that most of the sales happen in the months of November and December. This information is more easily available in column **MONTH_ID**. So I will be dropping the **ORDERDATE** column so as to decrease the risk of collinearity. Also the column **QTR_ID** seems to contain redundant information. So it will also be dropped.
 
 <h5 align="center">Line Plot of Sales V Date</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/line.png" width=600>
</p>
 
To analyze the frequency distribution of the dataset I plotted the below distplots for the following columns: **QUANTITYORDERED**, **SALES**, **PRICEEACH**, **MONTH_ID**, **MSRP**, **PRODUCTCODE**. 

<h5 align="center">Distplots of Features</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/dist1.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/dist2.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/dist3.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/dist4.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/dist5.png" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/dist6.png" width=600>
</p>

Next I use my favourite function in **Seaborn**, the _pairplot_ function. This plots all the columns against each other and it is easy to see any hidden pattern missed from just looking at heatmaps and correlation tables. From the below table I realized that:

1. There's a trend between **SALES** and **QUANTITYORDERED**
2. There's a trend between **MSRP** ad **PRICEEACH**
3. It seems that there is a growth in sales as years progress, which is a good sign for a company.

<h5 align="center">Pairplot of Features</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/pairplot.PNG" width=600>
</p>

## Clustering Model

KMeans clustering is one of the best clustering algorithms around. It uses the euclidean distances between the data points among the feature space to cluster data points accordingly. The only hyperparameter to set is the number of cluster parameters. This can be found by running a for loop to get the score of each cluster group and using the elbow method to determine the optimal number clusters. Below we can see that the elbow isn't very clear, but around cluster 5 seems to be where it is.

<h5 align="center">Elbow Plot of K Means</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/elbow1.PNG" width=600>
</p>

So we run the KMeans algorithm again to segment the full dataset into 5 clusters. Using the scaler inverse function on the cluster centres, gives us the values on how the clusters were formed. From the pic below we can say that each cluster has at least one defining trait seen by the scaler inverse table. These are:

1. **Cluster 0**: These customers buy items in high quantity at around 47 and they buy items in all price ranges but lean more towards the higher end. This group gives the highest total of all groups.
2. **Cluster 1**: This group represents customers who buy items in varying quantities around 35, they tend to buy high price items around 96. Their sales is bit better average ~4435, they buy products with second highest MSRP of around 133.
3. **Cluster 2**: This group represents customers who buy items in low quantity around 30. They tend to buy low price items around 68. Their sales around 2044, is lower than other clusters and they are extremely active around the holiday season. They buy products with low MSRP of around 75. 
4. **Cluster 3**: This group represents customers who are only active during the holidays. They buy in lower quantities around 35, but they tend to buy average price items around 86. They also correspond to lower total sales around 3673, they tend to buy items with MSRP around 102. 
5. **Cluster 4**:  This group represents customers who buy items in varying quantities around 39, they tend to buy average price items around 94. Their sales were around 4280. 

<h5 align="center">Table of Scaler Inverse Cluster Centres</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/centres1.PNG" width=600>
</p>

The above inferences can be made from the table above. The graphs below show the different columns in the dataset with respect to the cluster they are in. They match up with the inferences up above.

<h5 align="center">Features According to Clusters</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/col1.PNG" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/col2.PNG" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/col3.PNG" width=600>
</p>

To better understand how the clusters are formed, I will the 5 clusters in a 3-D space using **Plotly**. But since there are 39 feature columns in our dataset, it is not possible to visualize them in just a 3-D space without dimensionality reduction. So, I utilized the Principal Component Analysis (PCA) to reduce the dimensions from the original 39 to just 3. PCA is an unsupervised machine learning algorithm which reduces the dimensions of the data given but tries to keep the information unchanged. It does this by finding a new set of features called components which contain most of the information in the big feature space. The graph below is actually interactive thanks to **Plotly**, but README.md only allows static images.

<h5 align="center">3D Plot of Clusters</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/kmeans.png" width=600>
</p>

## Using AutoEncoder

I utilized Autoencoders to reduce the total number of features, while retaining the information, before applying a clustering algorithm. This led to a better performance by both K-Means and BIRCH clustering algorithms. Autoencoders do this by adding a bottleneck in the network, so this forces the network to compress whatever the input is given to it. This led to a simpler elbow method where the optimal number of clusters seem to be 3 not 5. This can be seen below:

<h5 align="center">Elbow Plot of K Means after Encoding</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/elbow2.PNG" width=600>
</p>

Applying the KMeans algorithm again on the reduced dataset we can see that the model performed much better. By performing the inverse transform on cluster centres, we can see that each cluster has a few defining traits. These traits can also be seen in the graphs below where all columns have been plotted with respect to the clusters.

1. **Cluster 0**: This group represents customers who buy items in high quantity(47), they usually buy items with high prices(99). They bring-in more sales than other clusters. They are mostly active through out the year. They usually buy products corresponding to product code 10-90. They buy products with high mrsp(158).
2. **Cluster 1**: This group represents customers who buy items in average quantity(37) and they buy tend to buy high price items(95). They bring-in average sales(4398) and they are active all around the year.They are the highest buyers of products corresponding to product code 0-10 and 90-100.Also they prefer to buy products with high MSRP(115).
3. **Cluster 2**: This group represents customers who buy items in small quantity(30), they tend to buy low price items(69). They correspond to the lowest total sale(2061) and they are active all around the year.They are the highest buyers of products corresponding to product code 0-20 and 100-110  they then to buy products with low MSRP(77).

<h5 align="center">Features According to Clusters</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/2col1.PNG" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/2col2.PNG" width=600>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/2col3.PNG" width=600>
</p>

I used PCA to again decrease the total dimensions of the data so it is easier to visualize. Below is the image of clusters by Kmeans:

<h5 align="center">3D Plot of Clusters</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/kmeans_auto.png" width=600>
</p>

Below is the picture of clusters by BIRCH:

<h5 align="center">3D Plot of Clusters: BIRCH</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Customer_Segmentation_Cluster/blob/main/Images/BIRCH.png" width=600>
</p>
 

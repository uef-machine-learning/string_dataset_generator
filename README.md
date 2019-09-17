# String Dataset Generator
Python3 scripts for string dataset generator for clustering purpose. 

*It is important to note thatThis implementation relies heavily on Jaccard similarity method.*

## How to run
### Create folder data
```sh
mkdir data
```
make sure you have your data file in there.

#### Example of content on data file
Let's say you have a data file containg a list of patients with icd-10 diseases.

The data file will look like the following

```
L01 K11 R44 X09 F00
F00 R21
M12 B20 L40 K50
.
.
.
W08 Q90 P00
```

### Execute the following command
```sh
# this will execute the example from main 
$ cd src/
$ python3 main.py
```

## Using it as library
To use this program as a library you can simple import `artificial_set_data_generator` in to your python module the call the following function `artificial_set_data_generator.generate`.

### The function takes the following parameters
* **data_size** : (int) an integer number specifies number of total number of data that will be generated
* **size_of_clusters** : (numpy arry) specifies size for each cluster. If empty array is passed then the size of all cluster will be the same.
                Note that len of array should equal to number_of_cluster and sum of this array should equal to data_size
* **number_of_cluster** : (int) an integer number specifies number of cluster to create
* **dimension** : (int) an integer number specifies total number of features that will be generate in the data set
* **distance_threshold** : (float) a number specifies the maximum distance away from the cluster representative according to Jaccard's method
* **size_of_set** : (tuple(int,int)) a tuple of intergers specifies the minimum and maximum feature that each data has to contain
* **all_features** : (string[]) an array of string containing all possible features of the dataset

## Output
From the `main.py` example, the output will be written the 3 separate files in out folder.

* gen_data.txt
* gen_representative.txt
* gen_ground_truths.txt

## Program flow chart
![program flowchart](./flowchart/flowchart.jpg)

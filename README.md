# string dataset generator
Python scripts for string dataset generator for clustering purpose. 

*It is important to note thatThis implementation relies heavily on Jaccard similarity method.*

## How to run
### Create folder data
```sh
mkdir data
```
make sure you have your data file in there.

#### example of content on data file
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

#### execute the following command
```sh
$ cd src/
$ python3 main.py
```

## Output
From the main.py example, the output will be written the 3 separate files in out folder.

* gen_data.txt
* gen_representative.txt
* gen_ground_truths.txt

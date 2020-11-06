# Anomaly-detection 
## Detecting anomalies using Z-score and TQR 

### Using Z score

	i>   finding mean and standard deviation of the dataset
	ii>  finding Z score, Z = (X-u)/sigma 
	iii> values for which Z > Threshold are outliers


### Using IQR:- 
	i>   interquantile range works on percentile
	ii>  sort the data in increasing order
        iii> let 25 and 75 percentile values be x and y respectively 
	iv>  finding IQR = y-x  
	ii>  values not in the range (x-1.5*IQR,y+1.5*IQR) are outlier.

## Detecting anomaly using LOF and Isolation Forest algorithms
## Detecting anomaly using neural network model 

### Note: 
	have a look at 'read_this' file

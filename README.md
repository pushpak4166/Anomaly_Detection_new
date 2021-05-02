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
### 
1) Regression_model.ipynb
	model:  regression_model.sav
	
2) Regression_model_2.ipynb 
	model:  regression_model_2.sav
		scalar_values_2.sav

3) Anomaly_using_NN_2.ipynb
	model:  anomaly_detection_model_final.h5 and .tflite 
	note: For predicting whether anomaly is present.

4) Anomaly_using_NN_new_improved.ipynb
	model:  anomaly_in_sensors_model.h5   
	accuracy: ~95%  
	note: this model is for detecting anomaly in each sensor(3 sensors)

5) Anomaly_detection_in_each_sensor.ipynb
	model: anomaly_in_4_sensors_model.h5  and .tflite
	accuracy: ~98%
	note: this model is for detecting anomaly in each sensor(4 sensors)

# Bouding-Box-Augmentastion-generator-script
Script to Augment images with bounding box data while training 

This script takes in an Excel file containing the path names and the bbox information, and uses that to 
augment the files while training as a datagenerator. 90% of the time the generator returns the normal image and does not agument it,
the other 10% it augments the image and returns the new points of the bounding box from the augmentation. This way the Network can
converge in a timley fashion while also having reistance to overfitting.

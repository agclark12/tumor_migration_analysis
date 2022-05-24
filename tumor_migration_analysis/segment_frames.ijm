//sets some initial parameters
sigma = 0.2;

//gets the image dimensions
run("Duplicate...", " ");
getDimensions(width, height, channels, slices, frames);

//goes through each frame
for (i=1; i<frames+1; i++) {

	//for each image, blur to reduce noise and threshold
	setSlice(i);
	run("Gaussian Blur...", "sigma="+sigma);
	
run("Auto Threshold", "method=Otsu white");

//cleans up the segmentation by binary operations
run("Open");
run("Fill Holes (Binary/Gray)");
run("Invert");
run("Watershed");
run("Invert");

//label the images by connected component analysis and view
run("Connected Components Labeling", "connectivity=4 type=[16 bits]");
	run("glasbey");
}
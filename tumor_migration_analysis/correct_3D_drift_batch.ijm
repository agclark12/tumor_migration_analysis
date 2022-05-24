setBatchMode(true);
dataDir = getDirectory("Choose Image Directory");
dirList = getFileList(dataDir);
saveDir = dataDir + "reg/";
File.makeDirectory(saveDir);
keyword = ".tif";

//goes through the directory list
for (i=0; i<lengthOf(dirList); i++) {

	fileName = dirList[i];
	if (matches(fileName, ".*" + keyword + ".*")) {

		//opens file
		open(dataDir + "/" + fileName);
		baseName = substring(filename, 0, lengthOf(filename) - 4);
		rename("stk");

		//corrects drift
		print("correcting drift");
		run("Correct 3D drift", "channel=1");

		selectWindow("registered time points");
		saveAs("tiff",saveDir + baseName + "_reg.tif");
		run("Close All");
		run("Collect Garbage");
	}
}
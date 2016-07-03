local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
cv.ml = require 'cv.ml'
local mnist = require 'mnist'
require 'optim'
torch.setdefaulttensortype('torch.FloatTensor')

-- Prepare mnist Data
local trainset = mnist.traindataset()
local testset = mnist.testdataset()
print(trainset.size) -- to retrieve the size
print(testset.size)  -- to retrieve the size
local trainData = trainset.data:reshape(60000, 784):float()
local trainLabel = (trainset.label+1):int()
local testData = testset.data:reshape(10000, 784):float()
local testLabel = (testset.label+1):int()
classes = {'1','2','3','4','5','6','7','8','9','0'}
confusion = optim.ConfusionMatrix(classes)

trainData = trainData[{ {1,200},{} }]
trainLabel = trainLabel[{ {1,200} }]
testData = testData[{ {1,200},{} }]
testLabel = testLabel[{ {1,200} }]

-- Build RT
-- Set up Random Tree's parameters
local rt = cv.ml.RTrees()
rt:setMaxDepth  		          {3}
rt:setMinSampleCount		      {1}
rt:setMaxCategories 		      {10}
rt:setCalculateVarImportance  {true}
rt:setRegressionAccuracy      {1e-4}
rt:setActiveVarCount          {2048}
rt:setTermCriteria            {cv.TermCriteria{cv.TermCriteria_MAX_ITER+cv.TermCriteria_EPS, 1000, 1e-4}}

-- Train the RT
local timer = torch.Timer()
rt:train{trainData, cv.ml.ROW_SAMPLE, trainLabel}
print("RT training time: " .. timer:time().real .. " seconds")
rt:save('./RT_MNIST_MODEL')

-- Test the RT
local predict = torch.Tensor(testData:size(1))
for i = 1, testData:size(1) do
	predict[i] = rt:predict{testData[i]}
	confusion:add(predict[i], testLabel[i])
end

-- Print confusion matrix
print(confusion)


local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
cv.ml = require 'cv.ml'
local mnist = require 'mnist'
torch.setdefaulttensortype('torch.FloatTensor')

-- Prepare mnist Data
local trainset = mnist.traindataset()
local testset = mnist.testdataset()
print(trainset.size) -- to retrieve the size
print(testset.size)  -- to retrieve the size
local trainData = trainset.data:reshape(60000, 784):float()
local trainLabel = (trainset.label + 1):int()
local testData = testset.data:reshape(10000, 784):float()
local testLabel = (testset.label + 1):int()

trainData = trainData[{ {1,10},{} }]
trainLabel = trainLabel[{ {1,10} }]

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
rt:train{trainData, cv.ml.ROW_SAMPLE, trainLabel}

-- Test the RT
local predict = torch.Tensor(trainData:size(1))
local timer = torch.Timer()
for i = 1, trainData:size(1) do
	predict[i] = rt:predict{trainData[i]}
end
print("RT evaluation time: " .. timer:time().real .. " seconds")

local accu = trainLabel - predict


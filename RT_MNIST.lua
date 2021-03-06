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

--[[
trainData = trainData[{ {1,1000},{} }]
trainLabel = trainLabel[{ {1,1000} }]
testData = testData[{ {1,500},{} }]
testLabel = testLabel[{ {1,500} }]
--]]

-- Build RT
-- Set up Random Tree's parameters
local rt = cv.ml.RTrees()
rt:setMaxDepth  		          {25}
rt:setMinSampleCount		      {5}
rt:setMaxCategories 		      {15}
rt:setCalculateVarImportance  {true}
rt:setRegressionAccuracy      {1e-2}
rt:setActiveVarCount          {4}
rt:setTermCriteria            {cv.TermCriteria{cv.TermCriteria_MAX_ITER+cv.TermCriteria_EPS+cv.TERM_CRITERIA_COUNT, 1000, 1e-2, 100}}

-- Train the RT
local timer = torch.Timer()
rt:train{trainData, cv.ml.ROW_SAMPLE, trainLabel}
print("RT training time: " .. timer:time().real .. " seconds")
print(rt:getActiveVarCount())
rt:save('./RT_MNIST_MODEL')

-- Test the RT
print(rt:getActiveVarCount())
local predict = torch.Tensor(testData:size(1))
for i = 1, testData:size(1) do
	predict[i] = rt:predict{testData[i]}
	confusion:add(predict[i], testLabel[i])
end
print(confusion)

-- Test the RT reload
local rt_new = cv.ml.RTrees()
rt_new:load('./RT_MNIST_MODEL')
confusion = optim.ConfusionMatrix(classes)
print(rt_new:getActiveVarCount())
local predict = torch.Tensor(testData:size(1))
for i = 1, testData:size(1) do
	predict[i] = rt_new:predict{testData[i]}
	confusion:add(predict[i], testLabel[i])
end
print(confusion)


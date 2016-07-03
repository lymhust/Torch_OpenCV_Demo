local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
cv.ml = require 'cv.ml'

-- Data for visual representation
local width, height = 512, 512
local im = torch.ByteTensor(height, width, 3):zero()

-- Set up training data
local labelsMat = torch.IntTensor{1, -1, -1, -1}
local trainingDataMat = torch.FloatTensor{ {501, 10}, {255, 10}, {501, 255}, {10, 501} }

-- Set up Random Tree's parameters
local rt = cv.ml.RTrees()
rt:setMaxDepth  		          {3}
rt:setMinSampleCount		      {1}
rt:setMaxCategories 		      {2}
rt:setCalculateVarImportance  {true}
rt:setRegressionAccuracy      {1e-4}
rt:setActiveVarCount          {2048}
rt:setTermCriteria            {cv.TermCriteria{cv.TermCriteria_MAX_ITER+cv.TermCriteria_EPS, 1000, 1e-4}}

-- Train the RT
rt:train{trainingDataMat, cv.ml.ROW_SAMPLE, labelsMat}

-- Show the decision regions given by the RT
local timer = torch.Timer()

local green, blue = torch.ByteTensor{0,255,0}, torch.ByteTensor{255,0,0}

for i=1,im:size(1) do
    for j=1,im:size(2) do
        local response = rt:predict{torch.FloatTensor{{j, i}}}
        im[{i,j,{}}]:copy(response == 1 and green or blue)
    end
end

print("RT evaluation time: " .. timer:time().real .. " seconds")

-- Show the training data
local thickness = -1
local lineType = 8
cv.circle{ im, {501,  10}, 5, {  0,   0,   0}, thickness, lineType }
cv.circle{ im, {255,  10}, 5, {255, 255, 255}, thickness, lineType }
cv.circle{ im, {501, 255}, 5, {255, 255, 255}, thickness, lineType }
cv.circle{ im, { 10, 501}, 5, {255, 255, 255}, thickness, lineType }

-- Show support vectors
thickness = 2
lineType  = 8

--cv.imwrite{"result.png", im}          -- save the image
cv.imshow{"RT Simple Example", im}    -- show it to the user
cv.waitKey{0}

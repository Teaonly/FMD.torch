require('nn')
require('image')

torch.setdefaulttensortype('torch.FloatTensor')

local classNumber = 21
local allBoxes = { {2,2}, {3,3}, {4,4}, {5,5},  
                   {2,4}, {4,2}, {3,6}, {6,3},
                   {2,6}, {6,2},
                   {2,3}, {3,2}, {4,6}, {6,4},
                   {2,5}, {5,2}, {3,4}, {4,3},
                   {3,5}, {5,3}, {4,5}, {5,4} }

local fixedCNN = torch.load('fixedCNN.t7');
fixedCNN:evaluate()

local featureCNN = nn.Sequential()

-- input is 56x56
featureCNN:add( nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil() )

-- input is 28x28
featureCNN:add( nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil() )

-- input is 14x14
featureCNN:add( nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
--[[
featureCNN:add( nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil() )
--]]
featureCNN:add(nn.SpatialConvolution(512, 1024, 1, 1, 1, 1, 0, 0))
featureCNN:add(nn.LeakyReLU(0.1))

local lossLayers = {}
local mbox = nn.ConcatTable()
for i = 1, #allBoxes do
    local boxConf = nn.Sequential()
    boxConf:add(nn.Dropout(0.5))
    boxConf:add(nn.SpatialConvolution(1024, classNumber, allBoxes[i][1], allBoxes[i][2], 1, 1, 0, 0))
    boxConf:add(nn.SpatialLogSoftMax())
    mbox:add(boxConf)
    table.insert(lossLayers, nn.SpatialClassNLLCriterion())
 
    local boxLoc = nn.Sequential()
    boxLoc:add(nn.SpatialConvolution(1024,  4, allBoxes[i][1], allBoxes[i][2], 1, 1, 0, 0))
    mbox:add(boxLoc)
    table.insert(lossLayers, nn.MSECriterion())  
end
featureCNN:add(mbox)

--[[
fixedCNN:cuda()
featureCNN:cuda()
local x = torch.rand(4, 3,256,256):cuda()
local y = featureCNN:forward( fixedCNN:forward(x) )
print(y)
--]]

local getSize = function(imageWidth, imageHeight) 
    local targetWidth = math.floor(imageWidth/16) 
    local targetHeight = math.floor(imageHeight/16)
    
    return {targetWidth, targetHeight};
end

local model = {}
model.fixedCNN = fixedCNN
model.featureCNN = featureCNN
model.lossLayers = lossLayers

local info = {}
info.classNumber = classNumber
info.boxes = allBoxes
info.getSize = getSize
model.info = info

return model

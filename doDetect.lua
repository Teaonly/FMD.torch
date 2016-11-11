require('torch')
require('cunn')
require('image')

local minSize = 224 
local threshold = 0.85

local className = {
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
}

local processImage = function(fileName)
    local img2caffe = function(img)
        local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
        local perm = torch.LongTensor{3, 2, 1}
        img = img:index(1, perm):mul(256.0)
        mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
        img:add(-1, mean_pixel)
        return img
    end

    local img = image.loadJPG(fileName)
    local wid = img:size()[3]
    local hei = img:size()[2]
   
    local scale = 0
    if ( wid > hei ) then
        scale = minSize / hei
    else
        scale = minSize / wid
    end
    local newWid = scale * wid
    local newHei = scale * hei
    
    if ( newWid < minSize) then 
        newWid = minSize
    end
    if ( newHei < minSize ) then
        newHei = minSize
    end
    
    newWid = newWid - (newWid % 32)
    newHei = newHei - (newHei % 32)
    
    local scaledImg = image.scale(img, newWid, newHei)
    local targetImg = img2caffe(scaledImg)

    return targetImg, scaledImg
end 

local jaccardOverlap = function(b1, b2)
    local xmin = math.min(b1.xmin, b2.xmin)
    local ymin = math.min(b1.ymin, b2.ymin)
    local xmax = math.max(b1.xmax, b2.xmax)
    local ymax = math.max(b1.ymax, b2.ymax)

    local w1 = b1.xmax - b1.xmin
    local h1 = b1.ymax - b1.ymin
    local w2 = b2.xmax - b2.xmin
    local h2 = b2.ymax - b2.ymin
  
    -- no overlap
    if ( (xmax - xmin) >= ( w1 + w2) or
        (ymax - ymin) >= ( h1 + h2) ) then
        return 0
    end
    
    local andw = w1 + w2 - (xmax - xmin)
    local andh = h1 + h2 - (ymax - ymin)

    local andArea = andw * andh
    local orArea = w1 * h1 + w2*h2 - andArea
    
    return andArea / orArea
end


-- init
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(2)

local _ = require('./model.lua')
local modelInfo = _.info
local fixedCNN = _.fixedCNN
local boxSampling = require('boxsampling')

local xinput, origin = processImage( arg[2]) 
_ = xinput:size()

local targetWidth = _[3]
local targetHeight = _[2]
local predBoxes = boxSampling( modelInfo, targetWidth, targetHeight)  

local featureCNN = torch.load(arg[1])
fixedCNN:cuda()
featureCNN:cuda()

local xfixed = fixedCNN:forward(xinput:cuda())
local yout = featureCNN:forward(xfixed)

local maxBoxes = {} 
local _ = modelInfo.getSize(targetWidth, targetHeight)
local lastWidth = _[1]
local lastHeight = _[2]

for i = 1, #modelInfo.boxes do
    local wid = lastWidth - (modelInfo.boxes[i][1] - 1)
    local hei = lastHeight - (modelInfo.boxes[i][2] - 1)
   
    local conf = yout[(i-1)*2 + 1]:float()
    local loc = yout[(i-1)*2 + 2]:float()
    
    for h = 1,hei do
        for w = 1,wid do
            local ii = "_" .. i .. "_" .. h .. "_" .. w
            local pbox = predBoxes[ii]
            local smf = conf[{{}, h, w}]:reshape(modelInfo.classNumber)
            local v,_ = smf:max(1)
            if ( _[1] ~= modelInfo.classNumber and math.exp(v[1]) > threshold ) then
                local box = {}
                box.v = math.exp(v[1])
                box.c = _[1]
                box.xmin = loc[1][h][w]*16 + pbox.xmin 
                box.ymin = loc[2][h][w]*16 + pbox.ymin
                box.xmax = loc[3][h][w]*16 + pbox.xmax
                box.ymax = loc[4][h][w]*16 + pbox.ymax
                
                local isNew = true
                local removed = {}
                for j = 1, #maxBoxes do
                    if ( maxBoxes[j].c == box.c ) then
                        overlap = jaccardOverlap(box, maxBoxes[j])
                        if ( overlap > 0.4) then
                            if ( box.v > maxBoxes[j].v ) then
                                table.insert(removed, j)
                            else
                                isNew = false
                                break
                            end
                        end
                    end
                end
                
                if isNew then
                    box.xmin = math.max(box.xmin, 0)
                    box.xmax = math.min(box.xmax, targetWidth)
                    box.ymin = math.max(box.ymin, 0)
                    box.ymax = math.min(box.ymax, targetHeight)
                    table.insert(maxBoxes, box)    
                    
                    for j = 1,#removed do
                        table.remove(maxBoxes, removed[j] - j + 1)    
                    end
                end
            end
        end
    end
end

print(maxBoxes)

local img = origin
for i = 1, #maxBoxes do
    img = image.drawRect(img, maxBoxes[i].xmin, maxBoxes[i].ymin, maxBoxes[i].xmax, maxBoxes[i].ymax)
    img = image.drawText(img, className[maxBoxes[i].c],  maxBoxes[i].xmin+10, (maxBoxes[i].ymin + maxBoxes[i].ymax) / 2)
end
image.save('result.jpg', img);
